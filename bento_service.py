from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, FileInput
from bentoml.service.artifacts.common import TextFileArtifact, JSONArtifact

from bentoml_artifacts.binary_artifact import BinaryFileArtifact
from bentoml_artifacts.onnx_artifact import OnnxModelArtifact


from typing import List, BinaryIO
import io
import json

from pre_post_processing import PrePostProcessing


@env(infer_pip_packages=True)
@artifacts([
    JSONArtifact('class_names'),
    TextFileArtifact('model_name'),
    OnnxModelArtifact('model', backend='onnxruntime'),
    JSONArtifact('config'),
    BinaryFileArtifact('test_image', file_extension='.jpg'),
    BinaryFileArtifact('test_result', file_extension=".pkl"),
    TextFileArtifact('service_version',
                     file_extension='.txt',
                     encoding='utf8') # for versions of bentoml 0.13 and newer 
])


class Model_YOLOX_Pack(BentoService):
    """    
    """    
    def __init__(self):
        super().__init__() 
        self.pre_post_processing = None
        self.config = None
        
    def set_prepost_processing(self):
        if not self.pre_post_processing:
            mean = self.config["Normalize"]["mean"]
            std = self.config["Normalize"]["std"]
            self.pre_post_processing = PrePostProcessing(mean=mean, std=std)
    
    def set_config(self):
        if not self.config:
            self.config = json.loads(self.artifacts.config)
    
    @api(input=JsonInput())
    def model_version(self, *args):
        """
        Run ID компонента, создавшего данную версию сервиса.
        """
        return self.artifacts.model_version

    @api(input=JsonInput())
    def service_version(self, *args):
        """
        Версия данного Bento сервиса.
        """
        return self.version
    
    @api(input=JsonInput())
    def class_names(self, *args):
        """
        Список классов детекции        
        """
        return self.artifacts.class_names
    
    @api(input=FileInput(), batch=False)
    def predict(self, file_stream):
        self.set_config()
        self.set_prepost_processing()
        input_data, scale_factors = self.pre_post_processing.prep_processing(file_stream)
        input_name = self.artifacts.model.get_inputs()[0].name
        output_name = [out.name for out in self.artifacts.model.get_outputs()]
        outs = self.artifacts.model.run(output_name, {input_name: input_data})        
        outs = self.pre_post_processing.post_processing(outs, scale_factors)
        return outs
    
    @api(input=JsonInput())
    def test_data(self, *args): # return some data for running a test
        return self.artifacts.test_image
        # pil_img=Image.open(self.artifacts.model_test_image)
        # image_array = np.asarray(pil_img)
        # return image_array

    
    @api(input=JsonInput(), batch=False)
    def test_result(self, *args): # return result data for a test data
        return self.artifacts.test_result
        # return pickle.load(io.BytesIO(bytearray(self.artifacts.test_result)))

