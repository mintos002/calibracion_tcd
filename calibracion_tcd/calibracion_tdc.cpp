#include "stdio.h"
#include <Windows.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"

#include "librealsense2/rs.hpp"

#pragma warning(disable : 4996)

typedef std::vector<std::string> stringvec;

// FUNCTIONS
void read_directory(const std::string& name, stringvec& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

std::string exePath()
{
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}

bool dirExists(const std::string& dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}

bool get_images_path(std::string path, stringvec& images)
{
	stringvec files;
	read_directory(path, files);
	std::string im = path;
	im += "\\";
	for (int i = 0; i < files.size(); i++)
	{
		
		std::size_t found_png = files[i].find(".png");
		std::size_t found_jpg = files[i].find(".jpg");
		if (found_png != std::string::npos || found_jpg != std::string::npos)
		{
			std::string p = im;
			p += files[i];
			images.push_back(p);
		}
		
	}
	if (files.size() == 0)
	{
		return false;
	}
	return true;
}


static double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints,
	const std::vector<std::vector<cv::Point2f> >& imagePoints,
	const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
	std::vector<float>& perViewErrors)
{
	std::vector<cv::Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);
		err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners/*,
	Settings::Pattern patternType*/ /*= Settings::CHESSBOARD*/)
{
	corners.clear();

	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(cv::Point3f(float((2 * j + i % 2)*squareSize), float(i*squareSize), 0));
}

static bool runCalibration(cv::Size boardSize, float squareSize, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
	std::vector<std::vector<cv::Point2f> > imagePoints, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
		std::vector<float>& reprojErrs, double& totalAvgErr)
{

	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

	distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

	std::vector<std::vector<cv::Point3f> > objectPoints(1);
	calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]/*, cv::CALIB_CB_ASYMMETRIC_GRID*/);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

	std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

// Print camera parameters to the output file
static void saveCameraParams(std::string outputFileName, cv::Size& t_imageSize,  cv::Mat& t_cameraMatrix,  cv::Mat& t_distCoeffs,
	const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
	const std::vector<float>& reprojErrs, const std::vector<std::vector<cv::Point2f> >& imagePoints,
	double totalAvgErr, cv::Size& c_imageSize, cv::Mat& c_cameraMatrix, cv::Mat& c_distCoeffs)
{
	cv::FileStorage fs(outputFileName, cv::FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm *t2 = localtime(&tm);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_Time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "thermal_image_Width" << t_imageSize.width;
	fs << "thermal_image_Height" << t_imageSize.height;
	fs << "color_image_Width" << c_imageSize.width;
	fs << "color_image_Height" << c_imageSize.height;

	fs << "Thermal_Camera_Matrix" << t_cameraMatrix;
	fs << "Thermal_Distortion_Coefficients" << t_distCoeffs;
	fs << "Color_Camera_Matrix" << c_cameraMatrix;
	fs << "Color_Distortion_Coefficients" << c_distCoeffs;

	fs << "Thermal_Avg_Reprojection_Error" << totalAvgErr;

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			cv::Mat r = bigmat(cv::Range(i, i + 1), cv::Range(0, 3));
			cv::Mat t = bigmat(cv::Range(i, i + 1), cv::Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		/*cvWriteComment(*fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0);
		fs << "Extrinsic_Parameters" << bigmat;*/
	}

	
}

bool runCalibrationAndSave(std::string outputFileName, cv::Size boardSize, float squareSize, cv::Size t_imageSize, cv::Mat&  t_cameraMatrix, cv::Mat& distCoeffs, std::vector<std::vector<cv::Point2f> > imagePoints, cv::Size& c_imageSize, cv::Mat& c_cameraMatrix, cv::Mat& c_distCoeffs)
{
	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(boardSize, squareSize, t_imageSize, t_cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
		reprojErrs, totalAvgErr);
	std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg re projection error = " << totalAvgErr;

	if (ok)
		saveCameraParams(outputFileName, t_imageSize, t_cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints, totalAvgErr, c_imageSize, c_cameraMatrix, c_distCoeffs);
	return ok;
}

// MAIN ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	const std::string inputSettingsFile = argc > 1 ? argv[1] : "configuration.xml";

	// CONFIG
	std::string path_thermo = "";
	std::string path_color = "";
	std::string out_config_name;
	int cd_resolution_mode = 0;
	int cd_imwidth = 1280;
	int cd_imheight = 720;
	int cd_fps = 30;
	int board_width;
	int board_height;
	float squareSize = 2.0;
	

	int flip_mode = 0;
	int negative_mode = 0;
	
	// READ CONFIG VALUES
	cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		printf("ERROR: Could not open the configuration file: %s\n", inputSettingsFile.c_str());
		system("pause");
		return -1;
	}
	
	fs["PathThermoImages"] >> path_thermo;
	fs["PathColorImages"] >> path_color;
	fs["OutputName"] >> out_config_name;
	fs["FlipMode"] >> flip_mode;
	fs["NegativeMode"] >> negative_mode;
	fs["BoardSize_Width"] >> board_width;
	fs["BoardSize_Height"] >> board_height;
	fs.release();

	// if no path defined, follow output_images folder structure
	if (path_thermo == "")
	{
		path_thermo = exePath();
		path_thermo += "\\output_images\\thermo";
	}
	if (path_color == "")
	{
		path_color = exePath();
		path_color += "\\output_images\\color";
	}
	if (out_config_name.empty())
	{
		printf("No output file name defined.\n");
		return -1;
	}
	switch (cd_resolution_mode) // Check resolution
	{
	case 0:
		cd_imwidth = 1280;
		cd_imheight = 720;
		break;
	case 1:
		cd_imwidth = 640;
		cd_imheight = 480;
		break;
	case 2:
		cd_imwidth = 848;
		cd_imheight = 480;
		break;
	default:
		printf("Resolution mode not recogniced. Set the resolution mode in the configuration file.\n");
		return -1;
		break;
	}
	if (cd_fps < 0) // Check fps
	{
		printf("Invalid FPS.\n");
		system("pause");
		return -1;
	}
	if (flip_mode < 0 && flip_mode > 1) // Check flip mode
	{
		printf("Invalid flip mode.\n");
		system("pause");
		return -1;
	}
	if (negative_mode < 0 && negative_mode > 1)
	{
		printf("Invalid flip mode.\n");
		system("pause");
		return -1;
	}
	if (board_width <= 0 || board_height <= 0)
	{
		printf("Invalid negative mode.\n");
		system("pause");
		return -1;
	}
	cv::Size patternsize(board_height, board_width);

	// Check if paths exists
	if (!dirExists(path_thermo))
	{
		printf("Path of thermal images do not exist.");
		system("pause");
		return -1;
	}
	
	// Variables
	stringvec thermo_images_path;
	stringvec color_images_path;

	// Check if images where found
	if (!get_images_path(path_thermo, thermo_images_path))
	{
		printf("No thermal images found inside the folder.");
		system("pause");
		return -1;
	}
	//// Check if images where found
	//if (!get_images_path(path_color, color_images_path))
	//{
	//	printf("No color images found inside the folder.");
	//	system("pause");
	//	return -1;
	//}

	// Configure intelrealsense
	rs2::pipeline pipe;
	rs2::config cfg;
	// Define transformations from and to Disparity domain
	rs2::decimation_filter dec;
	rs2::spatial_filter spat;
	rs2::temporal_filter temp;
	rs2::disparity_transform depth2disparity;
	rs2::disparity_transform disparity2depth(false);
	rs2::align align_to(RS2_STREAM_COLOR);
	rs2::frame depth_frame;
	cfg.enable_stream(RS2_STREAM_COLOR, cd_imwidth, cd_imheight, RS2_FORMAT_BGR8, cd_fps);
	cfg.enable_stream(RS2_STREAM_DEPTH, cd_imwidth, cd_imheight, RS2_FORMAT_Z16, cd_fps);

	// Start cameras
	auto prof = pipe.start(cfg);
	auto color_stream = prof.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
	auto depth_stream = prof.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	
	// Get color camera intrisincs
	auto c_intrinsics = color_stream.get_intrinsics();
	// Save realsense intrinsics
	cv::Mat c_cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	c_cameraMatrix.at<double>(0, 0) = c_intrinsics.fx;
	c_cameraMatrix.at<double>(1, 1) = c_intrinsics.fy;
	c_cameraMatrix.at<double>(0, 2) = c_intrinsics.ppx;
	c_cameraMatrix.at<double>(1, 2) = c_intrinsics.ppy;
	// Get distortion coefficients
	cv::Mat c_distCoeffs = cv::Mat::zeros(cv::Size(1, 5), CV_64F);
	c_distCoeffs.at<double>(1, 0) = c_intrinsics.coeffs[0];
	c_distCoeffs.at<double>(1, 1) = c_intrinsics.coeffs[1];
	c_distCoeffs.at<double>(1, 2) = c_intrinsics.coeffs[2];
	c_distCoeffs.at<double>(1, 3) = c_intrinsics.coeffs[3];
	c_distCoeffs.at<double>(1, 4) = c_intrinsics.coeffs[4];

	std::vector<std::vector<cv::Point2f> > imagePoints;
	cv::Mat t_cameraMatrix, t_distCoeffs;
	cv::Size t_imageSize, c_imageSize(cd_imwidth, cd_imheight);

	for (int i = 0;; i++)
	{
		cv::Mat frame;
		if (i < thermo_images_path.size())
		{
			frame = cv::imread(thermo_images_path[i], cv::IMREAD_ANYDEPTH);
		}
		

		if (frame.empty() || thermo_images_path.size() == i)          // If no more images then run calibration, save and stop loop.
		{
			if (imagePoints.size() > 0)
				printf("Starting camera calibration ...\n");
				bool ok = runCalibrationAndSave(out_config_name, patternsize, squareSize, t_imageSize, t_cameraMatrix, t_distCoeffs, imagePoints, c_imageSize, c_cameraMatrix, c_distCoeffs);
				if (ok) {
					cv::destroyAllWindows();
					bool showUndistort = false;
					if (showUndistort) 
					{
						for (int x = 0; x < thermo_images_path.size(); x++)
						{
							cv::Mat timg = cv::imread(thermo_images_path[x], cv::IMREAD_ANYDEPTH);
							cv::Mat cimg = cv::imread(thermo_images_path[x], cv::IMREAD_ANYDEPTH);
							// convert to 8bit
							timg.convertTo(timg, CV_8UC1, 1 / 256.0);

							cv::Mat tempt = timg.clone();
							cv::Mat tempc = cimg.clone();
							cv::undistort(tempt, timg, t_cameraMatrix, t_distCoeffs);
							cv::undistort(tempc, cimg, c_cameraMatrix, c_distCoeffs);

							cv::namedWindow("Undistort thermo", cv::WINDOW_AUTOSIZE);
							cv::imshow("Undistort thermo", timg);

							cv::namedWindow("Undistort color", cv::WINDOW_AUTOSIZE);
							cv::imshow("Undistort color", cimg);
							char key = (char)cv::waitKey(0);

							if (key == 27)
								break;
						}
					}
					
					
				}
				

			break;
		}

		t_imageSize = frame.size();

		//Image process
		cv::Mat t;
		if (flip_mode)
		{
			cv::flip(frame, t, 1);
			frame = t.clone();
		}
		if (negative_mode)
		{
			cv::bitwise_not(frame, frame);
		}

		// convert to 8bit
		frame.convertTo(frame, CV_8UC1, 1 / 256.0);
		
		std::vector<cv::Point2f> pointBuf;
		bool found;
		found = findCirclesGrid(frame, patternsize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID);

		if (found)
		{
			imagePoints.push_back(pointBuf);
			cv::drawChessboardCorners(frame, patternsize, cv::Mat(pointBuf), found);
		}

		if (frame.empty() == false)
		{
			cv::namedWindow("Thermal camera", cv::WINDOW_AUTOSIZE);
			cv::imshow("Thermal camera", frame);
		}
		/*system("pause");*/
		char key = (char)cv::waitKey(50);

		if (key == 27)
		{
			break;
		}
	}
	cv::destroyAllWindows();
	system("pause");
	return -1;
}