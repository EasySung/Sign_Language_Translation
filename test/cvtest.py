from tensorflow.keras.models import load_model

def test_load_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")

test_load_model('path/to/your/model.h5')
