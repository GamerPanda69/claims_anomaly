"""
Model Loader - Singleton pattern for loading pre-trained fraud detection models

TF-free shim: intercepts ALL pyod.models.auto_encoder and pyod.utils.torch_utility
class lookups via module-level __getattr__, so we never need to enumerate every
class name stored in the pickle.
"""
import os
import sys
import types
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Inject TF-free pyod shim BEFORE anything else (especially before joblib).
# We use module __getattr__ so ANY class the pickle requests is auto-created.
# ---------------------------------------------------------------------------

def _make_auto_stub_module(module_name: str):
    """
    Return a module whose __getattr__ creates a new stub class on first access.
    This handles any class the pickle references without us needing to list them.
    """
    mod = types.ModuleType(module_name)

    # Cache of auto-created stub classes
    _class_cache: Dict[str, type] = {}

    def _make_stub_class(class_name: str) -> type:
        """Dynamically create a stub class for any requested name."""
        if class_name in _class_cache:
            return _class_cache[class_name]

        # Try to make it a proper torch.nn.Module so weight restoration works
        try:
            import torch.nn as nn

            class _TorchStub(nn.Module):
                _stub_name = class_name

                def __init__(self, *args, **kwargs):
                    # Don't call super().__init__() here — pickle will call
                    # __setstate__ which calls nn.Module.__setstate__ properly
                    pass

                def __setstate__(self, state):
                    # nn.Module needs _parameters etc. initialised first
                    if '_parameters' not in state:
                        nn.Module.__init__(self)
                    self.__dict__.update(state)

                def forward(self, x):
                    # Standard pyod AutoEncoder structure
                    if hasattr(self, 'encoder') and hasattr(self, 'decoder'):
                        return self.decoder(self.encoder(x))
                    return x

            _TorchStub.__name__ = class_name
            _TorchStub.__qualname__ = class_name
            stub = _TorchStub

        except ImportError:
            # torch not available — plain object stub
            def _make_plain(name):
                cls = type(name, (), {
                    '__setstate__': lambda self, s: self.__dict__.update(s)
                })
                return cls
            stub = _make_plain(class_name)

        _class_cache[class_name] = stub
        return stub

    def _module_getattr(name: str):
        # Called when attribute not found normally on the module
        stub = _make_stub_class(name)
        setattr(mod, name, stub)   # cache on module so next lookup is O(1)
        return stub

    mod.__getattr__ = _module_getattr
    return mod


def _install_shim():
    """Force-inject fake pyod modules before any real pyod can be imported."""
    fake_ae = _make_auto_stub_module('pyod.models.auto_encoder')
    fake_tu = _make_auto_stub_module('pyod.utils.torch_utility')

    # Also add a hand-crafted AutoEncoder with proper predict() / decision_function()
    # so FraudDetector can call them after unpickling.
    class AutoEncoder:
        """Outer pyod detector stub — wraps the unpickled torch network."""

        def __setstate__(self, state: dict):
            self.__dict__.update(state)

        def _net(self):
            net = getattr(self, 'model_', None)
            if net is None:
                raise AttributeError("Unpickled AutoEncoder has no 'model_' attribute.")
            return net

        def predict(self, X):
            import torch
            net = self._net()
            net.eval()
            with torch.no_grad():
                t = torch.FloatTensor(np.asarray(X, dtype=np.float32))
                return net(t).numpy()

        def decision_function(self, X):
            X = np.asarray(X, dtype=np.float32)
            return np.mean(np.square(X - self.predict(X)), axis=1)

    # Register AutoEncoder explicitly; everything else is auto-created on demand
    fake_ae.AutoEncoder = AutoEncoder

    # Force-overwrite ONLY the two problematic leaf modules.
    # Do NOT touch pyod, pyod.models, or pyod.utils — the real pyod package
    # must remain importable so that pyod.models.iforest etc. work normally.
    sys.modules['pyod.models.auto_encoder'] = fake_ae
    sys.modules['pyod.utils.torch_utility'] = fake_tu


_install_shim()

# Now safe to import joblib
import joblib  # noqa: E402

logger = logging.getLogger(__name__)


def _joblib_load_cpu(path):
    """
    Load a joblib file while forcing all torch tensors onto CPU.

    The autoencoder was saved on a CUDA GPU. When joblib unpickles it,
    torch.load() is called internally without map_location, which crashes
    on CPU-only machines. We monkeypatch torch.load for the duration of
    the call to force map_location='cpu'.
    """
    import torch
    _original_torch_load = torch.load

    def _cpu_load(f, map_location=None, **kwargs):
        # Always override map_location to CPU
        return _original_torch_load(f, map_location=torch.device('cpu'), **kwargs)

    torch.load = _cpu_load
    try:
        return joblib.load(path)
    finally:
        torch.load = _original_torch_load  # always restore


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Singleton class to load and cache ML models"""

    _instance = None
    _models_loaded = False
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._models_loaded:
            self.load_models()

    def load_models(self):
        """Load all required models from the models directory"""
        models_dir = Path(os.getenv('MODELS_DIR', '/app/models'))

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        logger.info(f"Loading models from {models_dir}")

        required_models = {
            'autoencoder': 'autoenc.joblib',
            'iforest': 'iforest.joblib',
            'scaler': 'scaler.joblib',
            'clinical_norms': 'clinical_norms.joblib'
        }

        for model_name, filename in required_models.items():
            model_path = models_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Required model not found: {model_path}")

            try:
                self._models[model_name] = _joblib_load_cpu(model_path)
                logger.info(f"✓ Loaded {model_name} from {filename}")
            except Exception as e:
                logger.error(f"✗ Failed to load {model_name}: {e}")
                raise

        self._models_loaded = True
        logger.info("All models loaded successfully!")

    def get_model(self, model_name: str) -> Any:
        if not self._models_loaded:
            self.load_models()
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[model_name]

    @property
    def autoencoder(self):
        return self.get_model('autoencoder')

    @property
    def iforest(self):
        return self.get_model('iforest')

    @property
    def scaler(self):
        return self.get_model('scaler')

    @property
    def clinical_norms(self):
        return self.get_model('clinical_norms')

    def is_loaded(self) -> bool:
        return self._models_loaded


# Create singleton instance
model_loader = ModelLoader()
