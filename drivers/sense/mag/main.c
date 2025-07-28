/*
 * Copyright CogniPilot Foundation 2023
 * SPDX-License-Identifier: Apache-2.0
 */
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <math.h>

#include <cerebri/core/common.h>

#include <zros/private/zros_node_struct.h>
#include <zros/private/zros_pub_struct.h>
#include <zros/zros_node.h>
#include <zros/zros_pub.h>

#include <synapse_topic_list.h>

LOG_MODULE_REGISTER(sense_mag, CONFIG_CEREBRI_SENSE_MAG_LOG_LEVEL);

#define MY_STACK_SIZE 2048
#define MY_PRIORITY   6

extern struct k_work_q g_high_priority_work_q;
void mag_work_handler(struct k_work *work);

typedef struct ukf_magnetometer_state {
	// State vector θ' (9 elements)
	double c[3];              // Transformed bias vector
	double E[6];              // Transformed scale factor matrix [E11, E22, E33, E12, E13, E23]
	
	// Covariance matrix (9×9)
	double P[9][9];
	
	// UKF parameters
	double alpha;             // 0.1 (spread of sigma points)
	double beta;              // 2.0 (Gaussian distribution parameter)
	double kappa;             // 3 - L (secondary scaling)
	int L;                    // 9 (state dimension)
	
	// Working arrays
	double sigma_points[19][9];
	double weights_mean[19];
	double weights_cov[19];
} ukf_magnetometer_state_t;

typedef struct context {
	struct k_work work_item;
	const struct device *device[CONFIG_CEREBRI_SENSE_MAG_COUNT];
	struct zros_node node;
	struct zros_pub pub;
	synapse_pb_MagneticField data;
	
	// UKF state for real-time calibration
	ukf_magnetometer_state_t ukf_state;
	bool ukf_initialized;
	double geomagnetic_reference[3];
	double geomagnetic_magnitude;
} context_t;

static context_t g_ctx = {.work_item = Z_WORK_INITIALIZER(mag_work_handler),
			  .device = {},
			  .node = {},
			  .pub = {},
			  .data = {
				  .has_stamp = true,
				  .frame_id = "base_link",
				  .stamp = synapse_pb_Timestamp_init_default,
				  .magnetic_field = synapse_pb_Vector3_init_default,
				  .has_magnetic_field = true,
				  .magnetic_field_covariance = {},
				  .magnetic_field_covariance_count = 0,
			  },
			  .ukf_state = {
				  .c = {0.0, 0.0, 0.0},
				  .E = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0},
				  .alpha = 0.1,
				  .beta = 2.0,
				  .kappa = 0.0,
				  .L = 9
			  },
			  .ukf_initialized = false,
			  .geomagnetic_reference = {-1575.5, 19999.6, 48095.4},  // Typical values in nT
			  .geomagnetic_magnitude = 52111.0};

// Matrix helper functions
static void matrix_inverse_3x3(double A[3][3], double A_inv[3][3])
{
	double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
		     A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
		     A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
	
	if (fabs(det) < 1e-10) {
		// Identity matrix as fallback
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				A_inv[i][j] = (i == j) ? 1.0 : 0.0;
			}
		}
		return;
	}
	
	A_inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
	A_inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det;
	A_inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
	A_inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det;
	A_inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
	A_inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det;
	A_inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
	A_inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det;
	A_inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
}

static void cholesky_decomposition(double A[9][9], double L[9][9])
{
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			L[i][j] = 0.0;
		}
	}
	
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j <= i; j++) {
			if (i == j) {
				double sum = 0.0;
				for (int k = 0; k < j; k++) {
					sum += L[j][k] * L[j][k];
				}
				L[j][j] = sqrt(A[j][j] - sum);
			} else {
				double sum = 0.0;
				for (int k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
	}
}

// UKF helper functions
static void initialize_ukf(ukf_magnetometer_state_t *ukf)
{
	ukf->kappa = 3.0 - ukf->L;
	double lambda = ukf->alpha * ukf->alpha * (ukf->L + ukf->kappa) - ukf->L;
	
	// Initialize weights
	ukf->weights_mean[0] = lambda / (ukf->L + lambda);
	ukf->weights_cov[0] = ukf->weights_mean[0] + (1 - ukf->alpha * ukf->alpha + ukf->beta);
	
	for (int i = 1; i < 19; i++) {
		ukf->weights_mean[i] = 1.0 / (2.0 * (ukf->L + lambda));
		ukf->weights_cov[i] = ukf->weights_mean[i];
	}
	
	// Initialize covariance matrix
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			ukf->P[i][j] = (i == j) ? 1.0 : 0.0;
		}
	}
}

static void generate_sigma_points(ukf_magnetometer_state_t *ukf)
{
	double lambda = ukf->alpha * ukf->alpha * (ukf->L + ukf->kappa) - ukf->L;
	double gamma = sqrt(ukf->L + lambda);
	
	// Scale covariance matrix
	double scaled_P[9][9];
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			scaled_P[i][j] = (ukf->L + lambda) * ukf->P[i][j];
		}
	}
	
	// Cholesky decomposition
	double L[9][9];
	cholesky_decomposition(scaled_P, L);
	
	// Central sigma point
	for (int i = 0; i < 3; i++) {
		ukf->sigma_points[0][i] = ukf->c[i];
	}
	for (int i = 0; i < 6; i++) {
		ukf->sigma_points[0][i + 3] = ukf->E[i];
	}
	
	// Generate 2L sigma points
	for (int i = 0; i < ukf->L; i++) {
		for (int j = 0; j < ukf->L; j++) {
			ukf->sigma_points[i + 1][j] = ukf->sigma_points[0][j] + L[j][i];
			ukf->sigma_points[i + ukf->L + 1][j] = ukf->sigma_points[0][j] - L[j][i];
		}
	}
}

static double predict_measurement(double sigma_point[9], double B_k[3], double H_k[3])
{
	// Extract parameters from sigma point
	double c[3] = {sigma_point[0], sigma_point[1], sigma_point[2]};
	double E[6];
	for (int i = 0; i < 6; i++) {
		E[i] = sigma_point[i + 3];
	}
	
	// Form L_k vector for attitude-independent observation
	double L_k[9];
	L_k[0] = 2.0 * B_k[0];
	L_k[1] = 2.0 * B_k[1];
	L_k[2] = 2.0 * B_k[2];
	L_k[3] = -B_k[0] * B_k[0];
	L_k[4] = -B_k[1] * B_k[1];
	L_k[5] = -B_k[2] * B_k[2];
	L_k[6] = -2.0 * B_k[0] * B_k[1];
	L_k[7] = -2.0 * B_k[0] * B_k[2];
	L_k[8] = -2.0 * B_k[1] * B_k[2];
	
	// Compute L_k * θ'
	double prediction = 0.0;
	for (int i = 0; i < 3; i++) {
		prediction += L_k[i] * c[i];
	}
	for (int i = 0; i < 6; i++) {
		prediction += L_k[i + 3] * E[i];
	}
	
	return prediction;
}

static void convert_parameters(ukf_magnetometer_state_t *ukf, double b_out[3], double A_out[3][3])
{
	// Simplified parameter conversion - in practice you'd need eigenvalue decomposition
	// For now, use a simplified approach
	
	// Extract E matrix components
	double E_matrix[3][3] = {
		{ukf->E[0], ukf->E[3], ukf->E[4]},
		{ukf->E[3], ukf->E[1], ukf->E[5]},
		{ukf->E[4], ukf->E[5], ukf->E[2]}
	};
	
	// Simplified D calculation (assuming small perturbations)
	double D[3][3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j) {
				D[i][j] = sqrt(1.0 + E_matrix[i][j]) - 1.0;
			} else {
				D[i][j] = E_matrix[i][j] / 2.0;
			}
		}
	}
	
	// Compute (I + D)
	double I_plus_D[3][3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			I_plus_D[i][j] = (i == j ? 1.0 : 0.0) + D[i][j];
		}
	}
	
	// Compute inverse
	double I_plus_D_inv[3][3];
	matrix_inverse_3x3(I_plus_D, I_plus_D_inv);
	
	// b = (I + D)^(-1) * c
	for (int i = 0; i < 3; i++) {
		b_out[i] = 0.0;
		for (int j = 0; j < 3; j++) {
			b_out[i] += I_plus_D_inv[i][j] * ukf->c[j];
		}
	}
	
	// A = (I + D)^(-1)
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			A_out[i][j] = I_plus_D_inv[i][j];
		}
	}
}

static void ukf_update(ukf_magnetometer_state_t *ukf, double B_k[3], double H_k[3])
{
	// Generate sigma points
	generate_sigma_points(ukf);
	
	// Predict measurements for all sigma points
	double predicted_measurements[19];
	for (int i = 0; i < 19; i++) {
		predicted_measurements[i] = predict_measurement(ukf->sigma_points[i], B_k, H_k);
	}
	
	// Compute mean prediction
	double z_pred = 0.0;
	for (int i = 0; i < 19; i++) {
		z_pred += ukf->weights_mean[i] * predicted_measurements[i];
	}
	
	// Actual measurement (attitude-independent)
	double z_actual = 0.0;
	for (int i = 0; i < 3; i++) {
		z_actual += B_k[i] * B_k[i] - H_k[i] * H_k[i];
	}
	
	// Innovation
	double innovation = z_actual - z_pred;
	
	// Compute innovation covariance
	double S = 0.0;
	for (int i = 0; i < 19; i++) {
		double diff = predicted_measurements[i] - z_pred;
		S += ukf->weights_cov[i] * diff * diff;
	}
	S += 1.0; // Measurement noise
	
	// Compute cross-covariance
	double Pxz[9];
	for (int j = 0; j < 9; j++) {
		Pxz[j] = 0.0;
		for (int i = 0; i < 19; i++) {
			double x_diff = ukf->sigma_points[i][j] - ukf->sigma_points[0][j];
			double z_diff = predicted_measurements[i] - z_pred;
			Pxz[j] += ukf->weights_cov[i] * x_diff * z_diff;
		}
	}
	
	// Kalman gain
	double K[9];
	for (int i = 0; i < 9; i++) {
		K[i] = Pxz[i] / S;
	}
	
	// Update state
	for (int i = 0; i < 3; i++) {
		ukf->c[i] += K[i] * innovation;
	}
	for (int i = 0; i < 6; i++) {
		ukf->E[i] += K[i + 3] * innovation;
	}
	
	// Update covariance
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			ukf->P[i][j] -= K[i] * S * K[j];
		}
	}
}

void mag_work_handler(struct k_work *work)
{
	context_t *ctx = CONTAINER_OF(work, context_t, work_item);
	double mag_data_array[CONFIG_CEREBRI_SENSE_MAG_COUNT][3] = {};
	for (int i = 0; i < CONFIG_CEREBRI_SENSE_MAG_COUNT; i++) {
		// default all data to zero
		struct sensor_value mag_value[3] = {};

		// get accel if device present
		if (ctx->device[i] != NULL) {
			sensor_sample_fetch(ctx->device[i]);
			sensor_channel_get(ctx->device[i], SENSOR_CHAN_MAGN_XYZ, mag_value);
			LOG_DBG("mag %d: %d.%06d %d.%06d %d.%06d", i, mag_value[0].val1,
				mag_value[0].val2, mag_value[1].val1, mag_value[1].val2,
				mag_value[2].val1, mag_value[2].val2);
		}

		for (int j = 0; j < 3; j++) {
			mag_data_array[i][j] = mag_value[j].val1 + mag_value[j].val2 * 1e-6;
		}
	}

	// select first mag for data for now: TODO implement voting
	double mag[3] = {mag_data_array[0][1], -mag_data_array[0][0], -mag_data_array[0][2]};

	// Initialize UKF on first run
	if (!ctx->ukf_initialized) {
		initialize_ukf(&ctx->ukf_state);
		ctx->ukf_initialized = true;
		// LOG_INF("UKF magnetometer calibration initialized");
	}

	// Update UKF with new measurement
	ukf_update(&ctx->ukf_state, mag, ctx->geomagnetic_reference);

	// Extract current calibration parameters from UKF
	double b_estimated[3];
	double A_estimated[3][3];
	convert_parameters(&ctx->ukf_state, b_estimated, A_estimated);

	// Apply UKF-estimated calibration
	double temp[3];

	// Subtract estimated bias
	for (int i = 0; i < 3; i++){
		mag[i] -= b_estimated[i];
	}

	// Apply estimated calibration matrix
	for (int i = 0; i < 3; i++) {
		temp[i] = 0.0;
		for (int j = 0; j < 3; j++)
			temp[i] += A_estimated[i][j] * mag[j];
	}

	// Copy calibrated data back
	for (int i = 0; i < 3; i++){
		mag[i] = temp[i];
	}

	// Log calibration parameters periodically
	static int log_counter = 0;
	if (++log_counter >= 250) {  // Log every ~5 seconds at 50Hz
		LOG_INF("UKF bias: %.4f, %.4f, %.4f", b_estimated[0], b_estimated[1], b_estimated[2]);
		LOG_INF("UKF scale: %.4f, %.4f, %.4f", A_estimated[0][0], A_estimated[1][1], A_estimated[2][2]);
		log_counter = 0;
	}

	// publish
	stamp_msg(&ctx->data.stamp, k_uptime_ticks());
	ctx->data.magnetic_field.x = mag[0];
	ctx->data.magnetic_field.y = mag[1];
	ctx->data.magnetic_field.z = mag[2];
	zros_pub_update(&ctx->pub);
}

void mag_timer_handler(struct k_timer *dummy)
{
	k_work_submit_to_queue(&g_high_priority_work_q, &g_ctx.work_item);
}

K_TIMER_DEFINE(mag_timer, mag_timer_handler, NULL);

int sense_mag_entry_point(context_t *ctx)
{
	LOG_INF("init");
	ctx->device[0] = get_device(DEVICE_DT_GET(DT_ALIAS(mag0)));
#if CONFIG_CEREBRI_SENSE_MAG_COUNT >= 2
	ctx->device[1] = get_device(DEVICE_DT_GET(DT_ALIAS(mag1)));
#elif CONFIG_CEREBRI_SENSE_MAG_COUNT >= 3
	ctx->device[2] = get_device(DEVICE_DT_GET(DT_ALIAS(mag2)));
#elif CONFIG_CEREBRI_SENSE_MAG_COUNT >= 4
	ctx->device[3] = get_device(DEVICE_DT_GET(DT_ALIAS(mag3)));
#endif

	zros_node_init(&ctx->node, "sense_mag");
	zros_pub_init(&ctx->pub, &ctx->node, &topic_magnetic_field, &ctx->data);
	k_timer_start(&mag_timer, K_MSEC(20), K_MSEC(20));
	return 0;
}

K_THREAD_DEFINE(sense_mag, MY_STACK_SIZE, sense_mag_entry_point, &g_ctx, NULL, NULL, MY_PRIORITY, 0,
		100);

// vi: ts=4 sw=4 et
