#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <iostream>
#include <list>

#include "sift.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	//Mat img = imread(argv[1], IMREAD_GRAYSCALE);
	if (argc < 4)
	{
		cout << "ERROR: 3 arguments needed, @image1 path @ image2 path @ result image path" << endl;
		exit(1);
	}
	Mat img1_color = imread(argv[1]);
	Mat img2_color = imread(argv[2]);
	Mat img1_gray;
	Mat img2_gray;
	cvtColor(img1_color, img1_gray, CV_BGR2GRAY);
	cvtColor(img2_color, img2_gray, CV_BGR2GRAY);
	Mat f_img(img1_gray.size(), CV_64FC1);
	Mat f_img2(img2_gray.size(), CV_64FC1);
	img1_gray.convertTo(f_img, CV_64FC1);
	img2_gray.convertTo(f_img2, CV_64FC1);
	//imwrite("f_img.jpg", f_img);

	list<feature> feature_list1;
	list<feature> feature_list2;
	get_sift_descriptor(f_img, feature_list1);
	get_sift_descriptor(f_img2, feature_list2);
	cout << "image 1 feature size:" << feature_list1.size() << endl;
	cout << "image 2 feature size:" << feature_list2.size() << endl;
	match_feature(img1_color, img2_color, feature_list1, feature_list2, SIFT_DESCR_SIZE, argv[3]);
	//show_sift_desc(f_img, feature_list, argv[2]);
}