package com.cats.opencv.sample;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;

public class TrafficLineDetectionController {
	@FXML
	private Button browseButton;
	@FXML
	private TextField videoUrlText;
	@FXML
	private Button startButton;
	@FXML
	private ImageView originalFrame;
	@FXML
	private CheckBox fromVideoFile;
	@FXML
	private CheckBox fromCamera;

	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;

	private VideoCapture capture;
	// a flag to change the button behavior
	private boolean cameraActive;

	// cascade classifier
	private CascadeClassifier carCascade;
	private int absoluteCarSize;

	private static final String DEFAULT_VIDEO_FILE_URL = "resources/video/road.avi";

	private static final String DEFAULT_CAR_CASCADE_CLASSIFIER_PATH = "resources/haarcascades/haarcascade_cars3.xml";

	private String videoFileUrl = DEFAULT_VIDEO_FILE_URL;

	private boolean detectFromVideoFile = true;

	protected void init() {
		this.capture = new VideoCapture();
		this.carCascade = new CascadeClassifier();
		this.carCascade.load(DEFAULT_CAR_CASCADE_CLASSIFIER_PATH);
		this.absoluteCarSize = 0;
	}

	@FXML
	protected void startDetection() {
		// set a fixed width for the frame
		originalFrame.setFitWidth(600);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);

		if (!this.cameraActive) {
			// disable setting checkboxes
			this.fromVideoFile.setDisable(true);
			this.fromCamera.setDisable(true);

			// start the video capture
			if (detectFromVideoFile) {
				this.capture.open(videoFileUrl);// from video file
			} else {
				this.capture.open(0);// from camera
			}

			// is the video stream available?
			if (this.capture.isOpened()) {
				this.cameraActive = true;

				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {

					@Override
					public void run() {
						Image imageToShow = grabFrame();
						originalFrame.setImage(imageToShow);
					}
				};

				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

				// update the button content

				this.startButton.setText("Stop Camera");
			} else {
				System.err.println("Failed to open the camera connection...");
			}
		} else {
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.startButton.setText("Start Camera");
			this.fromVideoFile.setDisable(false);
			this.fromCamera.setDisable(false);

			try {
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}

			// release the camera
			this.capture.release();
			// clean the frame
			this.originalFrame.setImage(null);
		}
	}

	private Image grabFrame() {
		Image imageToShow = null;
		Mat frame = new Mat();

		if (this.capture.isOpened()) {
			try {
				// read the current frame
				this.capture.read(frame);

				// if the frame is not empty, process it
				if (!frame.empty()) {
					// detect cars and traffic lines
					this.detectAndDisplay(frame);
					// convert the Mat object (OpenCV) to Image (JavaFX)
					imageToShow = mat2Image(frame);
				}

			} catch (Exception e) {
				System.err.println("ERROR: " + e);
			}
		}

		return imageToShow;
	}
	
	private void detectRoad(Mat frame) {
		Mat gray = new Mat();
		Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
		Imgproc.threshold(gray, gray, 100, 255, Imgproc.THRESH_BINARY);
		
		Mat erode = new Mat();
		Imgproc.erode(gray, erode, new Mat(), new Point(2,2), 7);
		Mat dilate = new Mat();
		Imgproc.dilate(gray, dilate, new Mat(), new Point(2,2), 7);
		Imgproc.threshold(dilate, dilate, 1, 50, Imgproc.THRESH_BINARY_INV);
		
		Mat pathTrace = new Mat(gray.size(), CvType.CV_8U, new Scalar(0));
		Core.add(erode, dilate, pathTrace);
		
		Mat path = new Mat();
		pathTrace.convertTo(path, CvType.CV_32S);
		
		Imgproc.watershed(frame, path);
		path.convertTo(path, CvType.CV_8U);
		findAndDrawContours(path, frame);
	}
	

	private void detectAndDisplay(Mat frame) {
		MatOfRect cars = new MatOfRect();
		Mat grayFrame = new Mat();
		Mat distCanny = new Mat(frame.width(), frame.height(), frame.type());
		Mat halfFrame = new Mat(frame.width()/2, frame.height()/2,frame.type());
	    Imgproc.pyrDown(frame, halfFrame);
       

		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);

		// compute minimum car size (20% of the frame height, in our case)
		if (this.absoluteCarSize == 0) {
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0) {
				this.absoluteCarSize = Math.round(height * 0.2f);
			}
		}

		// detect cars
		this.carCascade.detectMultiScale(grayFrame, cars, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(50,
				50)/* new Size(this.absoluteCarSize, this.absoluteCarSize) */, new Size());
		System.out.println("face count is " + cars.size());
		// each rectangle in cars is a car: draw them!
		Rect[] carsArray = cars.toArray();
		for (int i = 0; i < carsArray.length; i++)
			Imgproc.rectangle(frame, carsArray[i].tl(), carsArray[i].br(), new Scalar(0, 255, 0), 3);

		// Crop off top half of image since we're only interested in the lower
		// portion of the video
		int halfHeight = frame.height() / 2;
		frame.locateROI(new Size(frame.width() - 1, halfHeight - 1), new Point(0, halfHeight));

		grayFrame.locateROI(new Size(frame.width() - 10, halfHeight - 10), new Point(0, halfHeight + 10));
		distCanny.locateROI(new Size(frame.width() - 10, halfHeight - 10), new Point(0, halfHeight + 10));

		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

		// Smooth images by remove noises
		Imgproc.GaussianBlur(grayFrame, grayFrame, new Size(3, 3), 5);

		//Improc.Canny(src, dst, lower_threshold, lower_threshld * ration = upper_threshold, kernel_size)
		Imgproc.Canny(grayFrame, distCanny, 50, 150, 5, true);

		Mat lines = new Mat();

		// Find lines
		Imgproc.HoughLinesP(distCanny, lines, 1, Math.PI / 180, 30, 200, 10);

		// Draw detected lines
		for (int x = 0; x < lines.cols(); x++) {
			for (int row = 0; row < lines.rows(); row++) {
				double[] vec = lines.get(row, x);
				System.out.println("Vec Length is " + vec.length);

				double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
				Point start = new Point(x1, y1);
				Point end = new Point(x2, y2);

				Imgproc.line(frame, start, end, new Scalar(255, 0, 0), 2);
			}
		}
		
		drawContours(frame);
		//detectRoad(frame);
		
		MatOfKeyPoint matOfKeyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.AKAZE);
		
		featureDetector.detect(frame, matOfKeyPoints);
		
		KeyPoint[] keyPoints = matOfKeyPoints.toArray();
		KeyPoint keyPoint1 = keyPoints[0];
		System.out.println(keyPoint1.angle+" key point angle");
		
		System.out.println(keyPoint1.pt);
		System.out.println(keyPoints.length+ " key point length");
		
		//Features2d.drawKeypoints(frame, matOfKeyPoints, frame, new Scalar(0, 0, 255), Features2d.DRAW_RICH_KEYPOINTS);
		
		drawContours2(frame);
		
		/*List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();

		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

		// Smooth images by remove noises
		Imgproc.GaussianBlur(grayFrame, grayFrame, new Size(3, 3), 5);

		//Improc.Canny(src, dst, lower_threshold, lower_threshld * ration = upper_threshold, kernel_size)
		Imgproc.Canny(grayFrame, distCanny, 50, 150, 5, true);
		// find contours
		Imgproc.findContours(distCanny, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

		// if any contour exist...
		if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
		{
		        // for each contour, display it in blue
		        for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
		        {
		                Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0));
		        }
		}*/

	}

	@FXML
	protected void fromVideoFileSelected(Event event) {
		if (this.fromCamera.isSelected())
			this.fromCamera.setSelected(false);

		this.detectFromVideoFile = true;
	}

	@FXML
	protected void fromCameraSelected(Event event) {
		if (this.fromVideoFile.isSelected())
			this.fromVideoFile.setSelected(false);

		this.detectFromVideoFile = false;
	}

	@FXML
	protected void browseVideoFile(Event event) {
		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Open Resource File");
		fileChooser.getExtensionFilters().addAll(new ExtensionFilter("Video Files", "*.avi", "*.mp4", "*.mpeg"));
		File selectedFile = fileChooser.showOpenDialog(null);
		if (selectedFile != null) {
			videoUrlText.setText(selectedFile.getAbsolutePath());
			videoFileUrl = selectedFile.getAbsolutePath();
		}
	}

	private Image mat2Image(Mat frame) {
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Imgcodecs.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}
	
	private void drawContours(Mat frame) {
		if (!frame.empty()) {
			frame.locateROI(new Size(frame.width()/2-1, frame.height()/2-1), new Point(0, frame.height()/2));
			// init
			Mat blurredImage = new Mat();
			Mat hsvImage = new Mat();
			Mat mask = new Mat();
			Mat morphOutput = new Mat();
			
			// remove some noise
			Imgproc.blur(frame, blurredImage, new Size(7, 7));
			
			// convert the frame to HSV
			Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HSV);
			
			// get thresholding values from the UI
			// remember: H ranges 0-180, S and V range 0-255
			Scalar minValues = new Scalar(0.0, 100.0, 56.0);
			Scalar maxValues = new Scalar(126.0,147.0, 180.0);
			
			// show the current selected HSV range
			//String valuesToPrint = "Hue range: " + minValues.val[0] + "-" + maxValues.val[0]
					//+ "\tSaturation range: " + minValues.val[1] + "-" + maxValues.val[1] + "\tValue range: "
					//+ minValues.val[2] + "-" + maxValues.val[2];
			//this.onFXThread(this.hsvValuesProp, valuesToPrint);
			
			// threshold HSV image to select tennis balls
			Core.inRange(hsvImage, minValues, maxValues, mask);
			// show the partial output
			//this.onFXThread(this.maskImage.imageProperty(), this.mat2Image(mask));
			
			// morphological operators
			// dilate with large element, erode with small ones
			Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(16, 16));
			Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8));
			
			Imgproc.erode(mask, morphOutput, erodeElement);
			Imgproc.erode(mask, morphOutput, erodeElement);
			
			Imgproc.dilate(mask, morphOutput, dilateElement);
			Imgproc.dilate(mask, morphOutput, dilateElement);
			
			// show the partial output
			//this.onFXThread(this.morphImage.imageProperty(), this.mat2Image(morphOutput));
			
			// find the tennis ball(s) contours and show them
			frame = this.findAndDrawContours(morphOutput, frame);
			
			// convert the Mat object (OpenCV) to Image (JavaFX)
		    mat2Image(frame);
		
		}
	}
	
	
	private void drawContours2(Mat frame) {
		Mat grayImage = new Mat();
		Mat thresholdImage = new Mat();
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		Imgproc.threshold(grayImage, thresholdImage,
				20, 255, Imgproc.THRESH_BINARY);
		Imgproc.blur(thresholdImage, thresholdImage, new Size(10, 10));
		Imgproc.threshold(grayImage, thresholdImage,
				20, 255, Imgproc.THRESH_BINARY);

		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Imgproc.findContours(thresholdImage, contours, hierarchy,
			Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		
		Rect objectBoundingRectangle = new Rect(0, 0, 0, 0);
		for (int i = 0; i < contours.size(); i++)
		{
			objectBoundingRectangle = Imgproc.boundingRect(contours.get(i));
			if(objectBoundingRectangle.area() > 50)
			Imgproc.rectangle(frame, objectBoundingRectangle.tl(), objectBoundingRectangle.br(), new Scalar(100,50,100));
		}
	}
	
	private Mat findAndDrawContours(Mat maskedImage, Mat frame)
	{
		// init
		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();
		
		// find contours
		Imgproc.findContours(maskedImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		
		for (int i=0; i < contours.size(); i++) {
			Rect boundingRectangle = Imgproc.boundingRect(contours.get(i));
			if (boundingRectangle.height> 20 && boundingRectangle.height< 100 &&
					boundingRectangle.width > 20 && boundingRectangle.width < 100) {
				Imgproc.rectangle(frame, new Point(boundingRectangle.x, boundingRectangle.y),new Point(boundingRectangle.x+ boundingRectangle.width-1, 
						boundingRectangle.y+boundingRectangle.height-1),   new Scalar(0,0,250),2);
			}
			
			
		}
		
		// if any contour exist...
		/*if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
		{
			// for each contour, display it in blue
			for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
			{
				//Imgproc.drawContours(frame, contours, idx, new Scalar(255, 0, 255), 3);
			}
		}*/
		
		return frame;
	}

}
