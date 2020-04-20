#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define INPUT_SIZE_FIX false
#define INPUT_WIDTH 2048
#define INPUT_HEIGHT 1024

Mat crop_window(Mat& image, int center_i, int center_j, int half_of_win, int window_size)
{
    unsigned char* image_data = image.data;
    Mat left_win = Mat::zeros(window_size, window_size, CV_8UC1);
    unsigned char* left_win_data = left_win.data;

    // crop window
    int win_start_row = center_i - half_of_win;
    int win_start_col = center_j - half_of_win;
    for(int win_i = 0; win_i < window_size; win_i++)
    {
        for(int win_j = 0; win_j < window_size; win_j++)
        {
            left_win_data[win_i*window_size + win_j] = image_data[(win_start_row + win_i)*image.cols + (win_start_col + win_j)];
        }
    }

    return left_win;
}

float fast_sqrt(const float& n)
{
   static union {int i; float f;} u;
   u.i = 0x2035AD0C + (*(int*)&n >> 1);
   return n / u.f + u.f * 0.25f;
}

float get_NCC(Mat& im1, Mat& im2)
{
    int length = im1.cols*im1.rows;
    unsigned char* im1_data = im1.data;
    unsigned char* im2_data = im2.data;

    float avg1, avg2;
    float omega1, omega2;
    float sum1 = 0, sum2 = 0;
    float ssd1 = 0, ssd2 = 0;
    for(int i = 0; i < length; i++)
    {
        sum1 += im1_data[i];
        sum2 += im2_data[i];
    }
    avg1 = sum1/float(length);
    avg2 = sum2/float(length);
    for(int i = 0; i < length; i++)
    {
        ssd1 += (float(im1_data[i]) - avg1)*(float(im1_data[i]) - avg1);
        ssd2 += (float(im2_data[i]) - avg2)*(float(im2_data[i]) - avg2);
    }
    omega1 = fast_sqrt(ssd1/float(length));
    omega2 = fast_sqrt(ssd2/float(length));

    float ncc;
    float tmp;
    for(int i = 0; i < length; i++)
    {
        tmp += ((float(im1_data[i]) - avg1)*(float(im2_data[i]) - avg2))/(omega1*omega2);
    }
    ncc = tmp/float(length);

    return ncc;
}

Mat stereo_block_match(Mat& left_image, Mat& right_image, int half_of_win)
{
    int window_size = half_of_win*2+1;
    Mat disp = Mat::zeros(left_image.rows, left_image.cols, CV_16SC1);
    short* disp_data = (short*)disp.data;

    if((left_image.type() != CV_8UC1) || (left_image.size() != right_image.size()))
        cout << "only can accept 8 bit grayscale image and same size" << endl;
    else
    {
        int width = left_image.cols;
        int height = left_image.rows;

        int start_row = half_of_win, end_row = height - 1 - half_of_win;
        int start_col = half_of_win, end_col = width - 1 - half_of_win;
        int search_length = width - half_of_win - half_of_win;

        #pragma omp parallel for collapse(2)
        for(int i = start_row; i <= end_row; i++)
        {
            for(int j = start_col; j <= end_col; j++)
            {
                Mat left_win = crop_window(left_image, i, j, half_of_win, window_size);

                vector<float> similarity(search_length, -1);
                int simil_idx = 0;
                for(int right_j = start_col; right_j <= j; right_j++)
                {
                    Mat right_win = crop_window(right_image, i, right_j, half_of_win, window_size);

                    similarity[simil_idx] = get_NCC(left_win, right_win);

                    simil_idx++;
                }
                
                std::vector<float>::iterator max_simil = std::max_element(similarity.begin(), similarity.end());
                int min_right_j = start_col + std::distance(similarity.begin(), max_simil);
                if(*max_simil > 0.5f) //good similarity
                    disp_data[i*width + j] = j - min_right_j;
                else if(*max_simil > 0.0f) //bad similarity
                    disp_data[i*width + j] = -1;
                else //zero or negative similarity
                    disp_data[i*width + j] = -2;
            }
        }

    }

    return disp;
}

int main(int argc, char* argv[])
{
    bool is_depth_gt = false;
    int half_of_win, numdisp;
    string left_name, right_name;

    if(argc != 5)
    {
        cout << "usage : filename.out <left_image> <right_image> <half_of_window> <number_of_disparity>" << endl;
        return 0;
    }
    else
    {
        left_name = argv[1];
        right_name = argv[2];
        half_of_win = stoi(argv[3]);
        numdisp = stoi(argv[4]);
    }

    Mat left_image = imread(left_name);
    Mat right_image = imread(right_name);

#if INPUT_SIZE_FIX == true
    resize(left_image, left_image, Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, INTER_CUBIC);
    resize(right_image, right_image, Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, INTER_CUBIC);
#endif

    Mat left_gray, right_gray;
    cvtColor(left_image, left_gray, CV_BGR2GRAY);
    cvtColor(right_image, right_gray, CV_BGR2GRAY);

    Mat disp = stereo_block_match(left_gray, right_gray, half_of_win);

    double minval, maxval;
    minMaxLoc(disp, &minval, &maxval);
    cout << "min: " << minval << " max: " << maxval << endl;

    string winname = "test";
    int thresh = 0;
    namedWindow(winname);
    createTrackbar("show threshold", winname, &thresh, 255);
    setTrackbarPos("show threshold", winname, 0);

    Mat bad_simil = (disp == -1);
    Mat negative_simil = (disp == -2);
    while(1)
    {
        Mat disp_mask = (disp > 0) & (disp < thresh);
        Mat disp_plus;
        disp.copyTo(disp_plus, disp_mask);
        Mat tmp;
        normalize(disp_plus, tmp, 0, 255, NORM_MINMAX, CV_8UC1);

        vector<Mat> tmp_arr = {bad_simil, tmp, negative_simil};
        Mat tmp_merge;
        merge(tmp_arr, tmp_merge);

        imshow(winname, tmp_merge);
        waitKey(10);
    }
}