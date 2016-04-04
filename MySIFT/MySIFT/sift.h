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
#define SIFT_EXTREMA_HARRIS 20
struct feature
{
	double x;                      /**< x coord */
	double y;                      /**< y coord */
	double scl;                    /**< scale of a Lowe-style feature */
	double ori_vec_angle;                    /**< orientation of a Lowe-style feature */
	double ori_vec_mag;
	double descr[FEATURE_MAX_D];   /**< descriptor */
	double im_size;
	feature(const double & x, const double &y, \
		const double & scl, const double im_size) :x(x), y(y), scl(scl), im_size(im_size)
	{
		
	}
};

int get_sift_descriptor(Mat & img, list<feature> &feature_list);
bool show_sift_desc(Mat & img, list<feature> &feature_list, const char out_file_name[]);
#endif //SIFT_H