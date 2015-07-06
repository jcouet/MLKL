/**************************************************************************************************/
/*                                                                                                */
/*  Informations :                                                                                */
/*      This code is part of the project MLKL                                                     */
/*                                                                                                */
/*  Contacts :                                                                                    */
/*      couet.julien@gmail.com                                                                    */
/*                                                                                                */
/**************************************************************************************************/

#include <vector>
#include <random>
#include <limits>
#include <time.h>
#include <string>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>  
#include <functional>
#include <type_traits>
using namespace std;

#include <MkCIFAR.h>
#include <FabricEDK.h>
using namespace Fabric::EDK;

IMPLEMENT_FABRIC_EDK_ENTRIES( MkCIFAR )


void ReadBatch(
  string filename,
  KL::VariableArray<KL::VariableArray<KL::Float64> > &images, 
  KL::VariableArray<KL::UInt32> &labels) 
{
  ifstream file (filename, ios::binary);
  if (file.is_open())
  {
    int number_of_images = 10000;
    int n_rows = 32;
    int n_cols = 32;

    double scale_max = 1.0;
    double scale_min = -1.0;

    images.resize(number_of_images);
    labels.resize(number_of_images);

    for(int i = 0; i < number_of_images; ++i)
    {
      unsigned char tplabel = 0;
      file.read((char*) &tplabel, sizeof(tplabel));
      labels[i] = KL::UInt32(tplabel);

      KL::VariableArray<KL::VariableArray<KL::Float64> > channels;
      channels.resize(3);
      channels[0].resize(n_rows*n_cols);
      channels[1].resize(n_rows*n_cols);
      channels[2].resize(n_rows*n_cols);

      images[i].resize(n_rows*n_cols*1);
      for(int ch = 0; ch < 3; ++ch)
      {
        KL::UInt32 count = 0;
        for(int r = 0; r < n_rows; ++r)
        {
          for(int c = 0; c < n_cols; ++c)
          {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            channels[ch][count] = KL::Float64(temp);
            //images[i][count] = KL::Float64(res);
            count++;
            //cerr << (int)temp << " " << res << endl;
          }
        }
      }

      KL::UInt32 count = 0;
      for(int r = 0; r < n_rows; ++r)
      {
        for(int c = 0; c < n_cols; ++c)
        {
          images[i][count] = (channels[0][count] + channels[1][count] + channels[2][count])/3.0;
          //images[i][count] = ((int)images[i][count] / 255.0) * (scale_max - scale_min) + scale_min;
          count ++;
        }
      }
    }
  }
}


FABRIC_EXT_EXPORT void MkCIFAR_parseBatch(
  KL::MkCIFAR::INParam expr,
  KL::String::INParam path,
  KL::VariableArray<KL::VariableArray<KL::Float64> >::IOParam images, 
  KL::VariableArray<KL::UInt32>::IOParam labels) 
{
  ReadBatch(string(path.data()), images, labels);
}
 
/*
#define elif else if
using namespace cv;
using namespace std;

Mat concatenateMat(vector<Mat> &vec) {

  int height = vec[0].rows;
  int width = vec[0].cols;
  Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
  for(int i=0; i<vec.size(); i++)
  {
    Mat img(height, width, CV_64FC1);
    Mat gray(height, width, CV_8UC1);
    cvtColor(vec[i], gray, CV_RGB2GRAY);
    gray.convertTo(img, CV_64FC1);
    // reshape(int cn, int rows=0), cn is num of channels.
    Mat ptmat = img.reshape(0, height * width);
    Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
    Mat subView = res(roi);
    ptmat.copyTo(subView);
  }
  divide(res, 255.0, res);
  return res;
}

Mat concatenateMatC(vector<Mat> &vec) {

  int height = vec[0].rows;
  int width = vec[0].cols;
  Mat res = Mat::zeros(height * width * 3, vec.size(), CV_64FC1);
  for(int i=0; i<vec.size(); i++)
  {
    Mat img(height, width, CV_64FC3);
    vec[i].convertTo(img, CV_64FC3);
    vector<Mat> chs;
    split(img, chs);
    for(int j = 0; j < 3; j++)
    {
      Mat ptmat = chs[j].reshape(0, height * width);
      Rect roi = cv::Rect(i, j * ptmat.rows, ptmat.cols, ptmat.rows);
      Mat subView = res(roi);
      ptmat.copyTo(subView);
    }
  }
  divide(res, 255.0, res);
  return res;
}

void read_CIFAR10(
  Mat &trainX, 
  Mat &testX, 
  Mat &trainY, 
  Mat &testY)
{
  string filename;
  filename = "cifar-10-batches-bin/data_batch_1.bin";
  vector<Mat> batch1;
  Mat label1 = Mat::zeros(1, 10000, CV_64FC1);    
  read_batch(filename, batch1, label1);

  filename = "cifar-10-batches-bin/data_batch_2.bin";
  vector<Mat> batch2;
  Mat label2 = Mat::zeros(1, 10000, CV_64FC1);    
  read_batch(filename, batch2, label2);

  filename = "cifar-10-batches-bin/data_batch_3.bin";
  vector<Mat> batch3;
  Mat label3 = Mat::zeros(1, 10000, CV_64FC1);    
  read_batch(filename, batch3, label3);

  filename = "cifar-10-batches-bin/data_batch_4.bin";
  vector<Mat> batch4;
  Mat label4 = Mat::zeros(1, 10000, CV_64FC1);    
  read_batch(filename, batch4, label4);

  filename = "cifar-10-batches-bin/data_batch_5.bin";
  vector<Mat> batch5;
  Mat label5 = Mat::zeros(1, 10000, CV_64FC1);    
  read_batch(filename, batch5, label5);

  filename = "cifar-10-batches-bin/test_batch.bin";
  vector<Mat> batcht;
  Mat labelt = Mat::zeros(1, 10000, CV_64FC1);    
  read_batch(filename, batcht, labelt);

  Mat mt1 = concatenateMat(batch1);
  Mat mt2 = concatenateMat(batch2);
  Mat mt3 = concatenateMat(batch3);
  Mat mt4 = concatenateMat(batch4);
  Mat mt5 = concatenateMat(batch5);
  Mat mtt = concatenateMat(batcht);

  Rect roi = cv::Rect(mt1.cols * 0, 0, mt1.cols, trainX.rows);
  Mat subView = trainX(roi);
  mt1.copyTo(subView);
  roi = cv::Rect(label1.cols * 0, 0, label1.cols, 1);
  subView = trainY(roi);
  label1.copyTo(subView);

  roi = cv::Rect(mt1.cols * 1, 0, mt1.cols, trainX.rows);
  subView = trainX(roi);
  mt2.copyTo(subView);
  roi = cv::Rect(label1.cols * 1, 0, label1.cols, 1);
  subView = trainY(roi);
  label2.copyTo(subView);

  roi = cv::Rect(mt1.cols * 2, 0, mt1.cols, trainX.rows);
  subView = trainX(roi);
  mt3.copyTo(subView);
  roi = cv::Rect(label1.cols * 2, 0, label1.cols, 1);
  subView = trainY(roi);
  label3.copyTo(subView);

  roi = cv::Rect(mt1.cols * 3, 0, mt1.cols, trainX.rows);
  subView = trainX(roi);
  mt4.copyTo(subView);
  roi = cv::Rect(label1.cols * 3, 0, label1.cols, 1);
  subView = trainY(roi);
  label4.copyTo(subView);

  roi = cv::Rect(mt1.cols * 4, 0, mt1.cols, trainX.rows);
  subView = trainX(roi);
  mt5.copyTo(subView);
  roi = cv::Rect(label1.cols * 4, 0, label1.cols, 1);
  subView = trainY(roi);
  label5.copyTo(subView);

  mtt.copyTo(testX);
  labelt.copyTo(testY);
}

int main() {
  Mat trainX, testX;
  Mat trainY, testY;
  trainX = Mat::zeros(1024, 50000, CV_64FC1);  
  testX = Mat::zeros(1024, 10000, CV_64FC1);  
  trainY = Mat::zeros(1, 50000, CV_64FC1);  
  testX = Mat::zeros(1, 10000, CV_64FC1);  

  read_CIFAR10(trainX, testX, trainY, testY);

  return 0;
}
*/