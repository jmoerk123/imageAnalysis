from imageAnalysis.src.detector import Detector, Viz

detector = Detector("KS")
viz = Viz(detector=detector, image="images/jacob.jpg")
viz.add_keypoints()
viz.add_keypoints_from_other_img("images/658934.png")
viz.save_image("test.png")
viz.show_image("Results")
