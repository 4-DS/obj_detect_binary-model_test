# Step CV-Pipeline: model_test

At this stage CV Pipeline Model_Test is executed:
 - extracting artifacts from the packaged BentoArchive (from the model_pack component);
 - testing the packed model's inference on a test image;
 - comparison of the result of the inferences of the packed model in BentoArchive and the model from the model_train component

Input data for step CV-Pipeline: model_test
- **coco_test_dataset**
  Dataset for detector testing from data_prep step
- **obj_detect_inference_files**
  Detector files from train step
- **bento_service**
  BentoArchive file from pack step
