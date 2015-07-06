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

#include <MkMNIST.h>
#include <FabricEDK.h>
using namespace Fabric::EDK;

IMPLEMENT_FABRIC_EDK_ENTRIES( MkMNIST )


#define ReverseEndian(x) ReverseEndianFunc((unsigned char *) &x, sizeof(x))
void ReverseEndianFunc(unsigned char * b, int n) {
  int i = 0;
  int j = n-1;
  while (i<j)
  {
    swap(b[i], b[j]);
    i++, j--;
  }
}

FABRIC_EXT_EXPORT void MkMNIST_parseLabels(
  KL::VariableArray<KL::UInt32>::IOParam labels, 
  KL::MkMNIST::INParam expr,
  KL::String::INParam path) 
{

  ifstream ifs(path.data(), ios::in | ios::binary);
  if (ifs.bad() || ifs.fail())
  {
    cerr << "Error MkMNIST_parseLabels : image-file error" << endl;
    return;
  }

  uint32_t magic_number;
  uint32_t num_items;
  ifs.read((char*) &magic_number, 4);
  ifs.read((char*) &num_items, 4);
  ReverseEndian(magic_number);
  ReverseEndian(num_items);
  if (magic_number != 0x00000801 || num_items <= 0)
  {
    cerr << "Error MkMNIST_parseLabels : image-file format error" << endl;
    return;
  }

  labels.resize(num_items);
  for(size_t i=0; i<num_items; i++) 
  {
    uint8_t label;
    ifs.read((char*) &label, 1);
    labels[i] = KL::UInt32(label);
  }
}

struct mnist_header {
  uint32_t magic_number;
  uint32_t num_items;
  uint32_t num_rows;
  uint32_t num_cols;
};

inline void parseMNISTImage(
  ifstream& ifs,
  const mnist_header& header,
  const KL::Float64 &scale_min,
  const KL::Float64 &scale_max,
  const KL::UInt32 &x_padding,
  const KL::UInt32 &y_padding,
  KL::VariableArray<KL::Float64> &dst) 
{
  const int width = header.num_cols + 2 * x_padding;
  const int height = header.num_rows + 2 * y_padding;

  vector<uint8_t> image_vec(header.num_rows * header.num_cols);
  ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);

  dst.resize(width * height);
  for (size_t i = 0; i < dst.size(); i++)
    dst[i] = scale_min;
  
  for (size_t y = 0; y < header.num_rows; y++)
  { 
    for (size_t x = 0; x < header.num_cols; x++)
      dst[width * (y + y_padding) + x + x_padding]
      = (image_vec[y * header.num_cols + x] / 255.0) * (scale_max - scale_min) + scale_min;
  }
}

inline void parseMNISTHeader(ifstream& ifs, mnist_header& header) {
  
  ifs.read((char*) &header.magic_number, 4);
  ifs.read((char*) &header.num_items, 4);
  ifs.read((char*) &header.num_rows, 4);
  ifs.read((char*) &header.num_cols, 4);
  ReverseEndian( header.magic_number);
  ReverseEndian( header.num_items);
  ReverseEndian( header.num_rows);
  ReverseEndian( header.num_cols);
  if (header.magic_number != 0x00000803 || header.num_items <= 0)
  {
    cerr << "Error parseMNISTHeader : image-file format error" << endl;
    return;
  }
  if (ifs.fail() || ifs.bad())
  {
    cerr << "Error parseMNISTHeader : file error" << endl;
    return;
  }
}

FABRIC_EXT_EXPORT void MkMNIST_parseImages(
  KL::VariableArray<KL::VariableArray<KL::Float64> >::IOParam images, 
  KL::MkMNIST::INParam expr,
  KL::String::INParam path,
  KL::Float64 scale_min,
  KL::Float64 scale_max,
  KL::Float64 x_padding,
  KL::Float64 y_padding) 
{
  ifstream ifs(path.data(), ios::in | ios::binary);
  if (ifs.bad() || ifs.fail())
    return;

  mnist_header header;
  parseMNISTHeader(ifs, header);

  images.resize(header.num_items);
  for (size_t i = 0; i < header.num_items; i++) 
    parseMNISTImage(ifs, header, scale_min, scale_max, x_padding, y_padding, images[i]);
}

/********/

template<typename T> inline 
typename enable_if<is_integral<T>::value, T>::type UniformRand(T min, T max) {
  static mt19937 gen(1);
  uniform_int_distribution<T> dst(min, max);
  return dst(gen);
}

template<typename T> inline 
typename enable_if<is_floating_point<T>::value, T>::type UniformRand(T min, T max) {
  static mt19937 gen(1);
  uniform_real_distribution<T> dst(min, max);
  return dst(gen);
}

template<typename T> void UniformRand(T *val, int size, T min, T max) {
  for (int i=0; i<size; ++i) 
    val[i] = UniformRand(min, max);
}
 
FABRIC_EXT_EXPORT void UniformRand_Float64( 
  KL::Traits< KL::Float64 >::INParam min, 
  KL::Traits< KL::Float64 >::INParam max,
  KL::Traits< KL::Float64 >::IOParam dist)
{
  dist = KL::Float64(UniformRand(double(min), double(max)));
}

FABRIC_EXT_EXPORT void UniformRand_Float32( 
  KL::Traits< KL::Float32 >::INParam min, 
  KL::Traits< KL::Float32 >::INParam max,
  KL::Traits< KL::Float32 >::IOParam dist)
{
  dist = KL::Float32(UniformRand(float(min), float(max)));
}

FABRIC_EXT_EXPORT void UniformRand_UInt32( 
  KL::Traits< KL::UInt32 >::INParam min, 
  KL::Traits< KL::UInt32 >::INParam max,
  KL::Traits< KL::UInt32 >::IOParam dist)
{
  dist = KL::UInt32(UniformRand(uint32_t(min), uint32_t(max)));
}

FABRIC_EXT_EXPORT void UniformRand_SInt32( 
  KL::Traits< KL::SInt32 >::INParam min, 
  KL::Traits< KL::SInt32 >::INParam max,
  KL::Traits< KL::SInt32 >::IOParam dist)
{
  dist = KL::SInt32(UniformRand(int(min), int(max)));
}

FABRIC_EXT_EXPORT void UniformRealDistribution_Float64( 
  KL::Traits< KL::Float64 >::INParam min, 
  KL::Traits< KL::Float64 >::INParam max,
  KL::Traits< KL::VariableArray<KL::Float64>>::IOParam dist)
{

  UniformRand(&dist[0], dist.size(), double(min), double(max));
}

FABRIC_EXT_EXPORT void UniformRealDistribution_Float32( 
  KL::Traits< KL::Float32 >::INParam min, 
  KL::Traits< KL::Float32 >::INParam max,
  KL::Traits< KL::VariableArray<KL::Float32>>::IOParam dist)
{

  UniformRand(&dist[0], dist.size(), float(min), float(max));
}

FABRIC_EXT_EXPORT void UniformRealDistribution_UInt32( 
  KL::Traits< KL::UInt32 >::INParam min, 
  KL::Traits< KL::UInt32 >::INParam max,
  KL::Traits< KL::VariableArray<KL::UInt32>>::IOParam dist)
{

  UniformRand(&dist[0], dist.size(), uint32_t(min), uint32_t(max));
}

FABRIC_EXT_EXPORT void UniformRealDistribution_SInt32( 
  KL::Traits< KL::SInt32 >::INParam min, 
  KL::Traits< KL::SInt32 >::INParam max,
  KL::Traits< KL::VariableArray<KL::SInt32>>::IOParam dist)
{

  UniformRand(&dist[0], dist.size(), int32_t(min), int32_t(max));
}

FABRIC_EXT_EXPORT void Bernoulli_Float64( 
  KL::Traits< KL::Float64 >::INParam p, 
  KL::Traits< KL::Boolean >::IOParam res)
{
  res = (UniformRand(0.0, 1.0) <= double(p));
}

FABRIC_EXT_EXPORT void Bernoulli_Float32( 
  KL::Traits< KL::Float32 >::INParam p, 
  KL::Traits< KL::Boolean >::IOParam res)
{
  res = (UniformRand(0.0f, 1.0f) <= float(p));
}

FABRIC_EXT_EXPORT void Bernoulli_Float64_UInt32( 
  KL::Traits< KL::Float64 >::INParam p, 
  KL::Traits< KL::UInt32 >::IOParam res)
{
  res = KL::UInt32(UniformRand(0.0, 1.0) <= double(p));
}

FABRIC_EXT_EXPORT void Bernoulli_Float32_UInt32( 
  KL::Traits< KL::Float32 >::INParam p, 
  KL::Traits< KL::UInt32 >::IOParam res)
{
  res = KL::UInt32(UniformRand(0.0f, 1.0f) <= float(p));
}
 
FABRIC_EXT_EXPORT void ReportR(KL::Traits< KL::String >::INParam str) {
  cerr << string(str.data()) << "\r";
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
FABRIC_EXT_EXPORT void CurrentDateTime(KL::Traits< KL::String >::IOParam str) {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tstruct);
  str = KL::String(string(buf).c_str());
}
