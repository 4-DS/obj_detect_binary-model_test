from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, FileInput
from bentoml.service.artifacts.common import TextFileArtifact, JSONArtifact

from sinara.bentoml_artifacts.binary_artifact import BinaryFileArtifact

from typing import List, BinaryIO
import io
import json


@env(infer_pip_packages=True)
@artifacts([
    TextFileArtifact('model_name'),
    BinaryFileArtifact('model', file_extension=".pth"),
    BinaryFileArtifact('config', file_extension=".py"),
    BinaryFileArtifact('test_image', file_extension='.jpg'),
    TextFileArtifact('service_version',
                     file_extension='.txt',
                     encoding='utf8') # for versions of bentoml 0.13 and newer 
])


class Model_YOLOX_Pack(BentoService):
    """    
    """ 
    
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
    def test_data(self, *args): # return some data for running a test
        return self.artifacts.test_image
    

