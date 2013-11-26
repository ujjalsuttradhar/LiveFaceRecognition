#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;


static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}



static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        path="/Users/ujjal/face/facerec/"+path;
        //cout<<path<<endl;
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));

            //Mat a=imread(path,CV_LOAD_IMAGE_COLOR);
            //namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
            //imshow( "Display window", a );

            labels.push_back(atoi(classlabel.c_str()));
        }
        //cout<<path<<endl;
    }
}


int main()
{
    string name;

    string output_folder;
    output_folder = "/Users/ujjal/face/facerec/output/";

    // Get the path to your CSV.
    string fn_csv = "/Users/ujjal/face/facerec/faces.txt";
    //Get the CascadeClassifier
    string fn_haar =  "/Users/ujjal/face/haarcascades/haarcascade_frontalface_alt_tree.xml";

    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    //cout<<images.size()<<endl;
    // Get the height from the first image. We'll need this
        // later in code to reshape the images to their original
        // size AND we need to reshape incoming faces to this size:
        int im_width = images[0].cols;
        int im_height = images[0].rows;


    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.

    /*Mat testSample = images[12];
    int testLabel = labels[12];
    images.pop_back();
    labels.pop_back();
*/


    //cvNamedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", images[1] );
    //waitKey(0);


    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      cv::createEigenFaceRecognizer(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0), call it with:
    //
    //      cv::createEigenFaceRecognizer(10, 123.0);
    //
    // If you want to use _all_ Eigenfaces and have a threshold,
    // then call the method like this:
    //
    //      cv::createEigenFaceRecognizer(0, 123.0);
    //

        //int num_components = 10;
        //double threshold = 10.0;

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
        // need to create the classifier for the task of Face Detection.
        // We are going to use the haar cascade you have specified in the
        // command line arguments:
        //
        CascadeClassifier haar_cascade;
        haar_cascade.load(fn_haar);
        // Get a handle to the Video device:
        //VideoCapture cap(deviceId);
        CvCapture *capture=cvCaptureFromCAM(0);
        // Check if we can use this device at all:
        //if(!cap.isOpened()) {
          //  cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
           // return -1;
        //}
        // Holds the current frame from the Video device:
        Mat frame;
        for(;;) {
            IplImage* iplImg = cvQueryFrame( capture );
                        frame = iplImg;
                        if( frame.empty() )
                            break;
            // Clone the current frame:
            Mat original = frame.clone();
            // Convert the current frame to grayscale:
            Mat gray;
            cvtColor(original, gray, CV_BGR2GRAY);
            // Find the faces in the frame:
            vector< Rect_<int> > faces;
            haar_cascade.detectMultiScale(gray, faces);
            // At this point you have the position of the faces in
            // faces. Now we'll get the faces, make a prediction and
            // annotate it in the video. Cool or what?
            for(int i = 0; i < faces.size(); i++) {
                // Process face by face:
                Rect face_i = faces[i];
                // Crop the face from the image. So simple with OpenCV C++:
                Mat face = gray(face_i);
                // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
                // verify this, by reading through the face recognition tutorial coming with OpenCV.
                // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
                // input data really depends on the algorithm used.
                //
                // I strongly encourage you to play around with the algorithms. See which work best
                // in your scenario, LBPH should always be a contender for robust face recognition.
                //
                // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
                // face you have just found:
                Mat face_resized;
                cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
                // Now perform the prediction, see how easy that is:
                int prediction = model->predict(face_resized);
                // And finally write all we've found out to the original image!
                // First of all draw a green rectangle around the detected face:
                rectangle(original, face_i, CV_RGB(0, 255,0), 1);
               // imshow("f", original);
               // char key = (char) waitKey(3);



                // Create the text we will annotate the box with:
                string box_text = name=format("Prediction = %d", prediction);
                // Calculate the position for annotated text (make sure we don't
                // put illegal values in there):
                int pos_x = std::max(face_i.tl().x - 10, 0);
                int pos_y = std::max(face_i.tl().y - 10, 0);
                // And now put it into the image:
                putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            }
            // Show the result:
            imshow("face_recognizer", original);
            // And display it:
            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if(key == 27)
                break;

        }

cout<<name<<endl;

    return 0;
}
