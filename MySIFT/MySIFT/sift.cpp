#include "sift.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <memory>
#include <time.h>
#include <stdlib.h>
#define PI 3.1415927
#define LITTLE 0.000001
//bool create_init_img(Mat & src_img, Mat & dst_img, double sigma);
int row = 5;
bool _create_gaus_pyr(Mat & base_img, Mat * Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int Octave_num, double *sigmas);
bool _get_extrema(Mat * DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], Mat *Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], list<feature> &feature_list, double * sigmas);
Mat * get_dog(Mat * Gaus_img, Mat * Next_Gaus_img);
bool _erase_dog_octaves(Mat *Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], int octave_num, int scale_num);
bool _erase_gaus_octaves(Mat *Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int octave_num, int scale_num);
bool _is_extremum(Mat * p_forward_DoG, Mat * p_this_DoG, Mat * p_Next_DoG, Mat * p_this_gaus_im, int r, int c);
int _calc_mag_angle(int bins_size, int x, int y, Mat * pGaussian_img, double & tmp_mag, double & angle);
bool _get_dog_pyr(Mat * Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], Mat * DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2]);
bool _get_accurate_pos(Mat * p_forward_DoG, Mat * p_this_DoG, Mat * p_Next_DoG, double & x, double & y, double sigmas[3]);
bool _get_feature_ori(feature & new_feature, Mat * pGaussian_img);
bool _is_this_too_edge(Mat * p_this_gaus_im, int r, int c);
int _calc_descriptor(Mat * Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3],\
	list<feature> & feature_list, double sigmas[SIFT_SCALE_DEGREE + 3]);
int _create_gauss_weight_mat(unsigned int mat_rows, unsigned int mat_cols, double * sigmas,\
	unsigned int size_sigma, Mat ** gauss_w_array);
int _normalize(double * descr, unsigned int sub_space_lenth, unsigned int sub_space_count);
shared_ptr<Mat> _combine_img(Mat & f_img, Mat & f_img2);
double _calc_similarity(double * p_descr1, double * p_descr2, int descr_size);
int _calc_mag_angle_d(int bins_size, int x, int y, Mat * pGaussian_img, double & tmp_mag);
Scalar _get_next_scalar();
bool _draw_sift_desc(Mat & img, list<feature> & feature_list)
{
	double p2_x = 0, p2_y = 0;
	double x = 0, y = 0;
	auto it = feature_list.begin();
	for (; it != feature_list.end(); ++it)
	{
		x = it->x * it->im_size;
		y = it->y * it->im_size;
		p2_x = x + it->ori_vec_mag / 10 * cos(it->ori_vec_angle);
		p2_y = y + it->ori_vec_mag / 10 * sin(it->ori_vec_angle);
		//cout << "scl:" << it->scl << endl;
		//cout << "im_size: " << it->im_size << endl;
		//arrowedLine(img, Point(x, y), Point(p2_x, p2_y), Scalar(255, 0, 0), 1);
		circle(img, Point(x, y), it->scl * it->im_size, Scalar(0, 255, 0), 1, LINE_AA);
	}
	return true;
}


Scalar _get_next_scalar()
{
	static bool is_first = true;
	if (is_first)
	{
		is_first = false;
		srand(time(NULL));
	}
	int B = rand() % 255;
	int G = rand() % 255;
	int R = rand() % 255;
	return Scalar(B, G, R);
}
double _calc_similarity(double * p_descr1, double * p_descr2, int descr_size)
{
	int i = 0;
	double simi = 0;
	for (i = 0; i < descr_size; ++i)
	{
		if (p_descr1[i] > 1)
		{
			cout << "bug" << endl;
		}
		if (p_descr2[i] > 1)
		{
			cout << "bug" << endl;
		}
		simi += p_descr1[i] * p_descr2[i];
	}
	if (simi < 0.0 || simi > 100)
	{
		cout << "bug" << endl;
	}
	//cout << simi << endl;
	return simi;
}
shared_ptr<Mat> _combine_img(Mat & f_img, Mat & f_img2)
{
	Size sz1 = f_img.size();
	Size sz2 = f_img2.size();
	shared_ptr<Mat> p_img3(new Mat(sz1.height, sz1.width + sz2.width, CV_8UC3));
	Mat left(*p_img3, Rect(0, 0, sz1.width, sz1.height));
	f_img.copyTo(left);
	Mat right(*p_img3, Rect(sz1.width, 0, sz2.width, sz2.height));
	f_img2.copyTo(right);
	return p_img3;
}
int match_feature(Mat & f_img, Mat & f_img2, list<feature> & feature_list1, list<feature> & feature_list2, int descr_size, const char out_file_name[])
{
	struct sort_match{
		bool operator()(const pair<double, pair<Point, Point> > & left, pair<double, pair<Point, Point> > & right)
		{
			return left.first > right.first;
		}
	};
	auto it1 = feature_list1.begin();
	auto it2 = feature_list2.begin();
	auto it_2_max = feature_list2.begin();
	double max_similarity = 0;
	double tmp_similarity = 0;
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	shared_ptr<Mat> pCombined_img = _combine_img(f_img, f_img2);
	vector<pair<double, pair<Point, Point> > > match_pairs;
	for (; it1 != feature_list1.end(); ++it1)
	{
		max_similarity = 0;
		x1 = it1->x * it1->im_size;
		y1 = it1->y * it1->im_size;
		for (it2 = feature_list2.begin(); it2 != feature_list2.end(); ++it2)
		{
			tmp_similarity = _calc_similarity(it1->descr, it2->descr, descr_size);
			if (max_similarity < tmp_similarity)
			{
				x2 = it2->x * it2->im_size;
				y2 = it2->y * it2->im_size;
				it_2_max = it2;
				max_similarity = tmp_similarity;
			}
		}
		if (x2 == 0)
		{
			cout << "bug" << endl;
		}
		if (y2 == 0)
		{
			cout << "bug" << endl;
		}
		x2 += f_img.cols;
		//arrowedLine(*pCombined_img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 1);
		match_pairs.push_back(pair<double, pair<Point, Point>>(max_similarity, pair<Point, Point>(Point(x1, y1), Point(x2, y2))));
	}
	sort(match_pairs.begin(), match_pairs.end(), sort_match());
	int i = 0;
	for (i = 0; i < SIFT_SHOW_AMOUNT; ++i)
	{
		cout << match_pairs[i].first << endl;
		arrowedLine(*pCombined_img, match_pairs[i].second.first, match_pairs[i].second.second, _get_next_scalar(), 3);
	}
	Mat left(*pCombined_img, Rect(0, 0, f_img.cols, f_img.rows));
	Mat right(*pCombined_img, Rect(f_img2.cols, 0, f_img2.cols, f_img2.rows));
	_draw_sift_desc(left, feature_list1);
	_draw_sift_desc(right, feature_list2);
	imwrite(out_file_name, *pCombined_img);
	waitKey(0);
	return 0;
}
int _normalize(double * descr, unsigned int sub_space_lenth, unsigned int sub_space_count)
{
	unsigned int i = 0;
	unsigned int j = 0;
	double sum = 0;
	for (i = 0; i < sub_space_count; ++i)
	{
		sum = 0;
		for (j = 0; j < sub_space_lenth; ++j)
		{
			sum += descr[i*sub_space_lenth + j] * descr[i*sub_space_lenth + j];
		}
		sum = sqrt(sum);
		if (sum == 0.0)
		{
			cout << "bug" << endl;
		}
		//if (descr[i*sub_space_lenth + j] > 1000)
		//{
		//	cout << "bug" << endl;
		//}
		for (j = 0; j < sub_space_lenth; ++j)
		{
			descr[i*sub_space_lenth + j] /= sum;
		}
	}
	for (i = 0; i < sub_space_count; ++i)
	{
		for (j = 0; j < sub_space_lenth; ++j)
		{
			if (descr[i*sub_space_lenth + j]> 1 || descr[i*sub_space_lenth + j] <0.0)
			{
				cout << "bug" << endl;
			}
		}
	}
	return 0;
}

int _create_gauss_weight_mat(unsigned int mat_rows, unsigned int mat_cols, double * sigmas,\
	unsigned int size_sigma, Mat ** gauss_w_array)
{
	unsigned int i = 0;
	unsigned int r = 0;
	unsigned int c = 0;
	double sigma = 0;
	double dr = 0;
	double dc = 0;
	double center_r = (mat_rows - 1) / 2;
	double center_c = (mat_cols - 1) / 2;
	assert(mat_rows > 0);
	assert(mat_cols > 0);
	assert(size_sigma > 1);
	for (i = 0; i < size_sigma; ++i)
	{
		sigma = sigmas[i];
		for (r = 0; r < mat_rows; ++r)
		{
			for (c = 0; c < mat_cols; ++c)
			{
				dr = abs(r - center_r);
				dc = abs(c - center_c);
				gauss_w_array[i]->at<double>(r, c) = exp(-(dr*dr + dc*dc)/ (2*sigma*sigma));
			}
		}
	}
	return 0;
}
int _calc_descriptor(Mat * Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], \
	list<feature> & feature_list, double sigmas[SIFT_SCALE_DEGREE + 3])
{
	// generate 18*18 neigbor matrix.
	Mat neib(Size(SIFT_DESCRIPTOR_MAT_SIZE + 2, SIFT_DESCRIPTOR_MAT_SIZE+2), CV_64F);
	// pre-create gaussian weight matrix.
	Mat * gauss_w_array[SIFT_SCALE_DEGREE + 2];
	int i = 0;
	int j = 0;
	for (i = 0; i < SIFT_SCALE_DEGREE + 2; ++i)
	{
		gauss_w_array[i] = new Mat(Size(SIFT_DESCRIPTOR_MAT_SIZE + 2, SIFT_DESCRIPTOR_MAT_SIZE + 2), CV_64F);
	}
	_create_gauss_weight_mat(SIFT_DESCRIPTOR_MAT_SIZE + 2, SIFT_DESCRIPTOR_MAT_SIZE + 2, sigmas,\
		SIFT_SCALE_DEGREE + 2, gauss_w_array);
	// process every feature point.
	auto it = feature_list.begin();
	double x0 = 0;
	double y0 = 0;
	int r = 0;
	int c = 0;
	double x = 0;
	double y = 0;
	double real_x = 0;
	double real_y = 0;
	double ori_angle = 0;
	int a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y;
	unsigned int oct = 0, im_id = 0;
	double grad_mag = 0;
	for (; it != feature_list.end(); ++it)
	{
		// get neib image for every sift descriptor point.
		x0 = it->x;
		y0 = it->y;
		ori_angle = it->ori_vec_angle;
		oct = it->gauss_octave_id;
		im_id = it->gauss_im_id;
		assert(im_id > 0);
		for (r = 0; r < SIFT_DESCRIPTOR_MAT_SIZE + 2; ++r)
		{
			for (c = 0; c < SIFT_DESCRIPTOR_MAT_SIZE + 2; ++c)
			{
				x = c - 8.5;
				y = r - 8.5;
				// transform the coordinate back to gauss image.
				real_x = x*cos(ori_angle) - y * sin(ori_angle) + x0;
				real_y = x*sin(ori_angle) + y * cos(ori_angle) + y0;
				if (real_y <= 0.5)
				{
					real_y = 0.51;
				}
				if (real_x <= 0.5)
				{
					real_x = 0.51;
				}
				// find the four neighbors.
				a_x = cvRound(real_x - 0.5);
				a_x = a_x >= Gaussian_Octaves[oct][im_id]->cols ? Gaussian_Octaves[oct][im_id]->cols - 1 : a_x;
				a_y = cvRound(real_y - 0.5);
				a_y = a_y >= Gaussian_Octaves[oct][im_id]->rows ? Gaussian_Octaves[oct][im_id]->rows - 1 : a_y;
				b_x = cvRound(real_x + 0.5);
				b_x = b_x >= Gaussian_Octaves[oct][im_id]->cols ? Gaussian_Octaves[oct][im_id]->cols - 1 : b_x;
				b_y = cvRound(real_y - 0.5);
				b_y = b_y >= Gaussian_Octaves[oct][im_id]->rows ? Gaussian_Octaves[oct][im_id]->rows - 1 : b_y;
				c_x = cvRound(real_x - 0.5);
				c_x = c_x >= Gaussian_Octaves[oct][im_id]->cols ? Gaussian_Octaves[oct][im_id]->cols - 1 : c_x;
				c_y = cvRound(real_y + 0.5);
				c_y = c_y >= Gaussian_Octaves[oct][im_id]->rows ? Gaussian_Octaves[oct][im_id]->rows -1 : c_y;
				d_x = cvRound(real_x + 0.5);
				d_x = d_x >= Gaussian_Octaves[oct][im_id]->cols ? Gaussian_Octaves[oct][im_id]->cols -1: d_x;
				d_y = cvRound(real_y + 0.5);
				d_y = d_y >= Gaussian_Octaves[oct][im_id]->rows ? Gaussian_Octaves[oct][im_id]->rows -1: d_y;
				// interpolate the pixel in (r, c)
				neib.at<double>(r, c) = (Gaussian_Octaves[oct][im_id]->at<double>(a_y, a_x) + \
					Gaussian_Octaves[oct][im_id]->at<double>(b_y, b_x) + Gaussian_Octaves[oct][im_id]->at<double>(c_y, c_x) + \
					Gaussian_Octaves[oct][im_id]->at<double>(d_y, d_x))/4;
			}
		}
		// get gradient hist of the SIFT_DESCRIPTOR_SUB_SPACE_WIDTH sub blocks.
		int sub_space_id = 0;
		int bin_id = 0;
		double tmp_mag = 0;
		double w = 0;
		for (r = 1; r <= SIFT_DESCRIPTOR_MAT_SIZE; ++r)
		{
			for (c = 1; c <= SIFT_DESCRIPTOR_MAT_SIZE; ++c)
			{
				sub_space_id = (r-1) / SIFT_DESCRIPTOR_SUB_SPACE_WIDTH * SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT + (c-1) / SIFT_DESCRIPTOR_SUB_SPACE_WIDTH;
				bin_id = _calc_mag_angle_d(SIFT_DESCRIPTOR_ORI_ANGLE_BIN, c, r, &neib, tmp_mag);
				w = gauss_w_array[im_id]->at<double>(r, c);
				it->descr[bin_id + sub_space_id * SIFT_DESCRIPTOR_ORI_ANGLE_BIN] += tmp_mag * w;
			}
		}
		_normalize(it->descr, SIFT_DESCRIPTOR_ORI_ANGLE_BIN, SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT * SIFT_DESCRIPTOR_SUB_SPACES_ROW_COUNT);
	}
	return 0;
}

bool _is_DoG_too_edge(Mat * p_this_DoG, int r, int c)
{
	double thr = 0;
	double index = 0;
	double this_val = p_this_DoG->at<double>(r, c);
	double dxx = p_this_DoG->at<double>(r, c + 1) + p_this_DoG->at<double>(r, c - 1) - 2 * this_val;
	double dyy = p_this_DoG->at<double>(r + 1, c) + p_this_DoG->at<double>(r - 1, c) - 2 * this_val;
	double dxy = p_this_DoG->at<double>(r + 1, c + 1) - p_this_DoG->at<double>(r + 1, c - 1) +\
						p_this_DoG->at<double>(r - 1, c + 1) - p_this_DoG->at<double>(r - 1, c - 1);
	dxy /= 4.0;
	double tr = dxx + dyy;
	double det = dxx* dyy - pow(dxy, 2);
	if (det < 0)
	{
		return true;
	}
	thr = pow(SIFT_HESSIAN_EDGE_THR + 1, 2) / SIFT_HESSIAN_EDGE_THR;
	index = pow(tr, 2) / det;

	if (index > thr)
	{
		return true;
	}
	else
	{
		return false;
	}
}
bool _is_this_too_edge(Mat * p_this_gaus_im, int r, int c)
{
	double thr = 0;
	double index = 0;
	double this_val = p_this_gaus_im->at<double>(r, c);
	double dxx = p_this_gaus_im->at<double>(r, c + 1) + p_this_gaus_im->at<double>(r, c - 1) - 2 * this_val;
	double dyy = p_this_gaus_im->at<double>(r + 1, c) + p_this_gaus_im->at<double>(r - 1, c) - 2 * this_val;
	double dxy = p_this_gaus_im->at<double>(r + 1, c + 1) - p_this_gaus_im->at<double>(r + 1, c - 1) +\
						p_this_gaus_im->at<double>(r - 1, c + 1) - p_this_gaus_im->at<double>(r - 1, c - 1);
	dxy /= 4.0;
	double tr = dxx + dyy;
	double det = dxx* dyy - pow(dxy, 2);
	if (det < 0)
	{
		return true;
	}
	thr = pow(SIFT_HESSIAN_EDGE_THR + 1, 2) / SIFT_HESSIAN_EDGE_THR;
	index = pow(tr, 2) / det;

	//if (index > thr || tr < SIFT_EXTREMA_HARRIS)
	//{
	//	return true;
	//}
	if (index > thr)
	{
		return true;
	}
	else
	{
		return false;
	}
}
bool show_sift_desc(Mat & img, list<feature> &feature_list, const char out_file_name[])
{
	double p2_x = 0, p2_y = 0;
	double x = 0, y = 0;
	auto it = feature_list.begin();
	for (; it != feature_list.end(); ++it)
	{
		x = it->x * it->im_size;
		y = it->y * it->im_size;
		p2_x = x + it->ori_vec_mag/10*cos(it->ori_vec_angle);
		p2_y = y + it->ori_vec_mag/10*sin(it->ori_vec_angle);
		//cout << "scl:" << it->scl << endl;
		//cout << "im_size: " << it->im_size << endl;
		arrowedLine(img, Point(x, y), Point(p2_x, p2_y), Scalar(255, 0, 0), 1);
		circle(img, Point(x, y), it->scl * it->im_size, Scalar(0, 255, 0), 1, LINE_AA);
	}
	imwrite(out_file_name, img);
	char order;
	cout << "print y to exit:" << endl;
	cin >> order;
	return true;
}
int get_sift_descriptor(Mat & img, list<feature> &feature_list)
{
	Mat * Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE+3];
	Mat * DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE+2];
	double sigmas[SIFT_SCALE_DEGREE + 3];
	double *p_sigmas = sigmas;
	double k = pow(2.0, 1.0 / SIFT_SCALE_DEGREE);
	for (int i = 0; i < SIFT_SCALE_DEGREE + 3; ++i)
	{
		sigmas[i] = SIFT_FIRST_SIGMA * pow(k, i - 1);
	}

	_create_gaus_pyr(img, Gaussian_Octaves, SIFT_OCTAVE, p_sigmas);
	_get_dog_pyr(Gaussian_Octaves, DoG_octaves);
	_get_extrema(DoG_octaves, Gaussian_Octaves, feature_list, sigmas);
	_calc_descriptor(Gaussian_Octaves, feature_list, sigmas);
	_erase_gaus_octaves(Gaussian_Octaves, SIFT_OCTAVE, SIFT_SCALE_DEGREE+3);
	_erase_dog_octaves(DoG_octaves, SIFT_OCTAVE, SIFT_SCALE_DEGREE+2);
	return 0;
}

bool _get_extrema(Mat * DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], Mat *Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], list<feature> &feature_list, double * sigmas)
{
	double x_extr = 0, y_extr = 0;
	double im_size = 0;
	for (int oct = 0; oct < SIFT_OCTAVE; ++oct)
	{
		for (int i = 1; i < SIFT_SCALE_DEGREE + 1; ++i)
		{
			for (int r = 0; r < DoG_octaves[oct][i]->rows; ++r)
			{
				for (int c = 0; c < DoG_octaves[oct][i]->cols; ++c)
				{
					if (_is_extremum(DoG_octaves[oct][i - 1], DoG_octaves[oct][i], DoG_octaves[oct][i + 1], Gaussian_Octaves[oct][i], r, c))
					{
						//Got an extrema
						x_extr = c;
						y_extr = r;
						if (false == _get_accurate_pos(DoG_octaves[oct][i-1], DoG_octaves[oct][i],\
														DoG_octaves[oct][i+1], x_extr, y_extr, sigmas+i-1))
						{
							// D(x,y) is too low.
							continue;
						}
						else
						{
							// record this new feature point.
							im_size = double(DoG_octaves[0][0]->cols) / double(DoG_octaves[oct][i]->cols);
							feature new_feature(x_extr, y_extr, sigmas[i], im_size, oct, i);
							// get the hist of this new feature.

							if (_get_feature_ori(new_feature, Gaussian_Octaves[oct][i]))
							{
								if (new_feature.x < 0.0 || new_feature.y < 0.0)
								{
									cout << "bug" << endl;
								}
								feature_list.push_back(new_feature);
							}
						}
						
					}
				}
			}
		}
	}
	return true;
}
bool _get_feature_ori(feature & new_feature, Mat * pGaussian_img)
{
	double x = new_feature.x, y = new_feature.y;
	double weight_sigma = SIFT_ORI_SIG_FCTR * new_feature.scl;
	double tmp_mag = 0;
	double exp_denom = 2.0 * weight_sigma * weight_sigma;
	double w = 0;
	double angle = 0;
	int range_radius = int(SIFT_ORI_RADIUS * new_feature.scl);
	int bin_id = 0;
	int i = -range_radius;
	int j = -range_radius;

	double p_hist_bins[SIFT_ORI_HIST_BINS] = { 0 };
	if (x < range_radius+1 || x > pGaussian_img->cols - range_radius-1 || y < range_radius+1 || y > pGaussian_img->rows - range_radius - 1)
	{
		return false; // this point is too close to the boundary of the image.
	}
	for (i = -range_radius; i < range_radius; ++i)
	{
		for (j = -range_radius; j < range_radius; ++j)
		{
			if (i > 10000)
			{
				throw;
			}
			bin_id = _calc_mag_angle(SIFT_ORI_HIST_BINS, int(x) + i, int(y) + j, pGaussian_img, tmp_mag, angle);
			w = exp(-(i*i + j*j) / exp_denom);
			p_hist_bins[bin_id] += tmp_mag * w;
		}
	}
	// find the max orientation.
	tmp_mag = p_hist_bins[0];
	bin_id = 0;
	for (int i = 1; i < SIFT_ORI_HIST_BINS; ++i)
	{
		if (p_hist_bins[i] > tmp_mag)
		{
			tmp_mag = p_hist_bins[i];
			bin_id = i;
		}
	}
	//new_feature.ori_vec_angle = (bin_id + 1) * 2 * PI / SIFT_ORI_HIST_BINS;
	new_feature.ori_vec_angle = angle;
	new_feature.ori_vec_mag = tmp_mag;
	if (tmp_mag < SIFT_MAG_THRES)
	{
		return false;
	}
	return true;
}
/* double version of _calc_mag_angle*/
int _calc_mag_angle_d(int bins_size, int x, int y, Mat * pGaussian_img, double & tmp_mag)
{
	double dx = 2.0 *pGaussian_img->at<double>(y, x + 1) + pGaussian_img->at<double>(y + 1, x + 1) + pGaussian_img->at<double>(y - 1, x + 1);
	dx -= 2.0 * pGaussian_img->at<double>(y, x - 1) + pGaussian_img->at<double>(y + 1, x - 1) + pGaussian_img->at<double>(y - 1, x - 1);
	double dy = 2.0 * pGaussian_img->at<double>(y + 1, x) + pGaussian_img->at<double>(y + 1, x + 1) + pGaussian_img->at<double>(y + 1, x - 1);
	dy -= 2.0 * pGaussian_img->at<double>(y - 1, x) + pGaussian_img->at<double>(y - 1, x + 1) + pGaussian_img->at<double>(y - 1, x - 1);
	tmp_mag = sqrt(dx*dx + dy*dy);
	tmp_mag += LITTLE;
	// get the angle of (dx, dy) vector.
	double angle = 0;
	int bin_id = 0;
	if (dy > 0)
	{
		angle = acos(dx / tmp_mag);
	}
	else
	{
		angle = 2 * PI - acos(dx / tmp_mag);
	}
	bin_id = int(angle / (2 * PI / bins_size));
	if (bin_id > bins_size || bin_id < 0)
	{
		throw;
	}
	return bin_id;
}
int _calc_mag_angle(int bins_size, int x, int y, Mat * pGaussian_img, double & tmp_mag, double & angle)
{
	double dx = 2.0 *pGaussian_img->at<double>(y, x + 1) + pGaussian_img->at<double>(y + 1, x + 1) + pGaussian_img->at<double>(y - 1, x + 1);
	dx -= 2.0 * pGaussian_img->at<double>(y, x - 1) + pGaussian_img->at<double>(y + 1, x - 1) + pGaussian_img->at<double>(y - 1, x - 1);
	double dy = 2.0 * pGaussian_img->at<double>(y + 1, x) + pGaussian_img->at<double>(y + 1, x + 1) + pGaussian_img->at<double>(y + 1, x - 1);
	dy -= 2.0 * pGaussian_img->at<double>(y - 1, x) + pGaussian_img->at<double>(y - 1, x + 1) + pGaussian_img->at<double>(y - 1, x - 1);
	tmp_mag = sqrt(dx*dx+dy*dy);
	tmp_mag += LITTLE;
	// get the angle of (dx, dy) vector.
	//double angle = 0;
	int bin_id = 0;
	if (dy > 0)
	{
		angle = acos(dx / tmp_mag);
	}
	else
	{
		angle = 2*PI - acos(dx / tmp_mag);
	}
	bin_id = int(angle / (2 * PI / bins_size));
	if (bin_id > bins_size || bin_id < 0)
	{
		throw;
	}
	return bin_id;
}
/*
* Calculate the accurate position fo the extrema. And eliminate it if the D(x, y) is too low.
*/
bool _get_accurate_pos(Mat * p_forward_DoG, Mat * p_this_DoG, Mat * p_Next_DoG, double & x, double & y, double sigmas[3])
{
	double dx = (p_this_DoG->at<double>(y, x+1) - p_this_DoG->at<double>(y, x-1)) / 2.0;
	double dy = (p_this_DoG->at<double>(y+1, x) - p_this_DoG->at<double>(y-1, x))/ 2.0;
	double d_theta = (p_forward_DoG->at<double>(y, x) - p_Next_DoG->at<double>(y, x)) / 2.0;
	double D_1diff[3][1] = { { dx }, { dy }, { d_theta } };
	Mat D1_Mat(3, 1, CV_64F, D_1diff);
	double this_val = p_this_DoG->at<double>(y, x);
	double dxx = p_this_DoG->at<double>(y, x + 1) + p_this_DoG->at<double>(y, x - 1) - 2 * this_val;
	double dxy = p_this_DoG->at<double>(y + 1, x + 1) - p_this_DoG->at<double>(y + 1, x - 1) - \
		p_this_DoG->at<double>(y - 1, x + 1) + p_this_DoG->at<double>(y - 1, x - 1);
	dxy /= 4.0;
	double dx_theta = (p_forward_DoG->at<double>(y, x + 1) - p_forward_DoG->at<double>(y, x - 1) - \
		p_Next_DoG->at<double>(y, x + 1) + p_forward_DoG->at<double>(y, x - 1)) / 2 / (*sigmas - *(sigmas + 2));
	double dyy = p_this_DoG->at<double>(y + 1, x) + p_this_DoG->at<double>(y - 1, x) - 2 * this_val;
	double dy_theta = (p_forward_DoG->at<double>(y+1, x) - p_forward_DoG->at<double>(y-1, x) -\
		p_Next_DoG->at<double>(y + 1, x) + p_Next_DoG->at<double>(y - 1, x)) / 2 / (*sigmas - *(sigmas + 2));
	double d_theta_theta = (p_forward_DoG->at<double>(y, x) - this_val) / (*sigmas - *(sigmas + 1)) -\
		(this_val - p_Next_DoG->at<double>(y, x))/(*(sigmas+2) - *(sigmas+1));
	d_theta_theta /= (*sigmas - *(sigmas + 2))/2;

	double D_2diff[3][3] = { { dxx, dxy, dx_theta }, { dxy, dyy, dy_theta }, {dx_theta, dy_theta, d_theta_theta} };
	Mat D2_Mat(3, 3, CV_64F, D_2diff) ;
	Mat D1_trsp;
	transpose(D1_Mat, D1_trsp);
	Mat deta = -D1_trsp * D2_Mat.inv();
	if (norm(deta) > 0.5)
	{
		this_val = this_val + 0.5 * D1_trsp.dot(deta);
	}
	if (this_val < SIFT_DoG_THR)
	{
		return false;// discard this point.
	}
	// Change the coordination to the accurate one.
	//x += deta.at<double>(0, 0);
	//y += deta.at<double>(0, 1);
	//if (x < 0.0 && y < 0.0)
	//{
	//	cout << "bug" << endl;
	//}
	// transfer (x,y) from Pyramid coordination to image coordination.
	//x *= im_size;
	//y *= im_size;
	return true;
}

bool _get_dog_pyr(Mat * Gaussian_Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], Mat * DoG_octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2])
{
	for (int i = 0; i < SIFT_OCTAVE; ++i)
	{
		for (int j = 1; j < SIFT_SCALE_DEGREE + 3; ++j)
		{
			DoG_octaves[i][j - 1] = get_dog(Gaussian_Octaves[i][j - 1], Gaussian_Octaves[i][j]);
		}
	}
	return true;
}

Mat * get_dog(Mat * Gaus_img, Mat * Next_Gaus_img)
{
	Mat * pDoG = new Mat(Gaus_img->size(), CV_64F);
	if (pDoG == nullptr)
	{
		throw;
	}
	//MatIterator_<double> it1 = Gaus_img->begin<double>(), end1 = Gaus_img->end<double>();
	//MatIterator_<double> it2 = Next_Gaus_img->begin<double>(), end2 = Gaus_img->endl<double>();
	//MatIterator_<double> it_dot = pDoG->begin<double>();
	//while (it1 != end1)
	//{

	//}
	subtract(*Gaus_img, *Next_Gaus_img, *pDoG);
	return pDoG;
}

bool _is_extremum(Mat * p_forward_DoG, Mat * p_this_DoG, Mat * p_Next_DoG, Mat * p_this_gaus_im, int r, int c)
{
	int dx = 0;
	int dy = 0;
	double max_abs = 0;
	if (r <= 0 || r >= p_this_DoG->rows - 1)
	{
		return false;
	}
	if (c <= 0 || c >= p_this_DoG->cols - 1)
	{
		return false;
	}
	bool is_max = true, is_min = true;
	double this_val = p_this_DoG->at<double>(r, c);
	// check if it is a maximum.
	if (this_val >= 0)
	{
		for (dx = -1; dx <= 1; ++dx)
		{
			for (dy = -1; dy <= 1; ++dy)
			{
				if (abs(p_forward_DoG->at<double>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<double>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<double>(r + dy, c + dx) > this_val)
				{
					return false;
				}
			}

		}
		for (dx= -1; dx <= 1; ++dx)
		{
			for (dy = -1; dy <= 1; ++dy)
			{
				if (dx == 0 && dy == 0)
				{
					continue;
				}
				if (abs(p_forward_DoG->at<double>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<double>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<double>(r + dy, c + dx) > this_val)
				{
					return false;
				}
			}
		}

	}
	// check if it is a minimum.
	else if (this_val < 0)
	{
		for (int dx = -1; dx <= 1; ++dx)
			for (int dy = -1; dy <= 1; ++dy)
			{
				if (abs(p_forward_DoG->at<double>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<double>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<double>(r + dy, c + dx) < this_val)
				{
					return false;
				}
			}

		for (int dx = -1; dx <= 1; ++dx)
			for (int dy = -1; dy <= 1; ++dy)
			{
				if (dx == 0 && dy == 0)
				{
					continue;
				}
				if (abs(p_forward_DoG->at<double>(r + dy, c + dx) - this_val) > max_abs)
				{
					max_abs = p_forward_DoG->at<double>(r + dy, c + dx) - this_val;
				}
				if (p_forward_DoG->at<double>(r + dy, c + dx) < this_val)
				{
					return false;
				}
			}
	}
	//if (max_abs < EXTREMA_DIST_THR)
	//{
	//	return false;
	//}
	//Check if it is a point in edge.
	if (_is_DoG_too_edge(p_this_gaus_im, r, c))
	{
		return false;
	}
	else
	{
		return true;
	}
	//if (_is_this_too_edge(p_this_gaus_im, r, c))
	//{
	//	return false;
	//}
	//else
	//{
	//	return true;
	//}

	//double dxx = p_this_DoG->at<double>(r, c + 1) + p_this_DoG->at<double>(r, c - 1) - 2 * this_val;
	//double dyy = p_this_DoG->at<double>(r + 1, c) + p_this_DoG->at<double>(r - 1, c) - 2 * this_val;
	//double dxy = p_this_DoG->at<double>(r + 1, c + 1) - p_this_DoG->at<double>(r + 1, c - 1) +\
	//				p_this_DoG->at<double>(r - 1, c + 1) - p_this_DoG->at<double>(r - 1, c - 1);
	//dxy /= 4.0;
	//double tr = dxx + dyy;
	//double det = dxx* dyy - pow(dxy, 2);
	//if (det < 0)
	//{
	//	return false;
	//}
	//thr = pow(SIFT_HESSIAN_EDGE_THR + 1, 2) / SIFT_HESSIAN_EDGE_THR;
	//index = pow(tr, 2) / det;

	//if (index > thr)
	//{
	//	return false;
	//}
	//else
	//{
	//	return true;
	//}

}

bool _erase_dog_octaves(Mat *Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 2], int octave_num, int scale_num)
{
	for (int i = 0; i < octave_num; ++i)
	{
		for (int j = 0; j < scale_num; ++j)
		{
			delete Octaves[i][j];
		}
	}
	return true;
}
bool _erase_gaus_octaves(Mat *Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int octave_num, int scale_num)
{
	for (int i = 0; i < octave_num; ++i)
	{
		for (int j = 0; j < scale_num; ++j)
		{
			delete Octaves[i][j];
		}
	}
	return true;
}

bool _create_gaus_pyr(Mat & base_img, Mat * Octaves[SIFT_OCTAVE][SIFT_SCALE_DEGREE + 3], int Octave_num, double *sigmas)
{

	Mat copy_base_im(base_img);

	for (int i = 0; i < Octave_num; i++)
	{
		if (i > 0)
		{
			//copy_base_im.resize(copy_base_im.rows / 2);
			pyrDown(copy_base_im, copy_base_im, Size(copy_base_im.cols / 2, copy_base_im.rows / 2));
		}
		for (int j = 0; j < SIFT_SCALE_DEGREE + 3; ++j)
		{
			Octaves[i][j] = new Mat(copy_base_im.rows, copy_base_im.cols, CV_64F);
			if (Octaves[i][j] == nullptr)
			{
				throw;
			}
			GaussianBlur(copy_base_im, *Octaves[i][j], Size(0, 0), sigmas[j], 0);
		}
	}
	return true;
}
//bool create_init_img(Mat & src_img, Mat & dst_img, double sigma)
//{
//
//}