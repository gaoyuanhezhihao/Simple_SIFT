#ifndef SIFT_H
#define SIFT_H
#include "opencv2\core\core.hpp"
#include <list>
using namespace std;
using namespace cv;

/** default sigma for initial gaussian smoothing */
#define SIFT_FIRST_SIGMA 1.6
#define SIFT_OCTAVE 6
#define SIFT_SCALE_DEGREE 3 
#define SIFT_HESSIAN_EDGE_THR 3
#define SIFT_DoG_THR 3
/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36
/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR
#define FEATURE_MAX_D 128
#define EXTREMA_DIST_THR 0.1
#define SIFT_EXTREMA_HARRIS 0.5
#define SIFT_DESCRIPTOR_MAT_SIZE 16
#define SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT 2 // subspace size if 2*2 blocks.
#define SIFT_DESCRIPTOR_SUB_SPACE_SIZE (SIFT_DESCRIPTOR_MAT_SIZE/ SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT)
#define SIFT_DESCRIPTOR_ORI_ANGLE_BIN 8
#define SIFT_DESCR_SIZE (SIFT_DESCRIPTOR_ORI_ANGLE_BIN * SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT * SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT)
#define SIFT_SHOW_AMOUNT 50
#define SIFT_MAG_THRES 1
struct feature
{
	double x;                      /**< x coord */
	double y;                      /**< y coord */
	double scl;                    /**< scale of a Lowe-style feature */
	double ori_vec_angle;                    /**< orientation of a Lowe-style feature */
	double ori_vec_mag;
	double descr[FEATURE_MAX_D];   /**< descriptor */
	int desc_size{ 32 };
	double im_size;
	int gauss_octave_id;
	int gauss_im_id;
	feature(const double & x, const double &y, \
		const double & scl, const double im_size,\
		int gauss_octave_id, int gauss_im_id) :\
		x(x), y(y), scl(scl), im_size(im_size), gauss_octave_id(gauss_octave_id),\
		gauss_im_id(gauss_im_id)
	{
		int i = 0;
		for (i = 0; i < FEATURE_MAX_D; ++i)
		{
			descr[i] = 0;
		}
	}
};

int get_sift_descriptor(Mat & img, list<feature> &feature_list);
bool show_sift_desc(Mat & img, list<feature> &feature_list, const char out_file_name[]);
int match_feature(Mat & f_img, Mat & f_img2, list<feature> & feature_list1, list<feature> & feature_list2, int descr_size, const char out_file_name[]);
#endif //SIFT_H