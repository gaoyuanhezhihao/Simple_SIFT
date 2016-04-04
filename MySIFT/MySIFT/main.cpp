#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <list>

#include "sift.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	//Mat img = imread(argv[1], IMREAD_GRAYSCALE);
	Mat img = imread(argv[1], IMREAD_GRAYSCALE);
	Mat f_img(img.size(), CV_32FC1);
	img.convertTo(f_img, CV_32FC1);
	//imwrite("f_img.jpg", f_img);

	list<feature> feature_list;
	get_sift_descriptor(f_img, feature_list);
	cout << "feature size:" << feature_list.size() << endl;
	show_sift_desc(f_img, feature_list, argv[2]);
}