import streamlit as st
import tensorflow as tf
import numpy as np
import h5py
import os

# Constants
ECG_REST_LEADS = {
    'strip_I': 0, 'strip_II': 1, 'strip_III': 2, 'strip_V1': 3, 'strip_V2': 4, 'strip_V3': 5,
    'strip_V4': 6, 'strip_V5': 7, 'strip_V6': 8, 'strip_aVF': 9, 'strip_aVL': 10, 'strip_aVR': 11,
}
ECG_SHAPE = (5000, 12)
ECG_HD5_PATH = 'ukb_ecg_rest'
print(os.getcwd())
MODEL_PATH = 'ml4h/model_zoo/ECG2AF/ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5'

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load the ECG2AF model from the specified path.

    Args:
        model_path (str): Path to the .h5 model file.

    Returns:
        tf.keras.Model: Loaded TensorFlow model.
    """
    return tf.keras.models.load_model(model_path)

def ecg_as_tensor(ecg_file: str) -> np.ndarray:
    """
    Convert an ECG file to a tensor.

    Args:
        ecg_file (str): Path to the ECG file in HD5 format.

    Returns:
        np.ndarray: Tensor representation of the ECG data.
    """
    with h5py.File(ecg_file, 'r') as hd5:
        tensor = np.zeros(ECG_SHAPE, dtype=np.float32)
        for lead in ECG_REST_LEADS:
            data = np.array(hd5[f'{ECG_HD5_PATH}/{lead}/instance_0'])
            tensor[:, ECG_REST_LEADS[lead]] = data
        tensor -= np.mean(tensor)
        tensor /= np.std(tensor) + 1e-6
    return tensor

def process_ecg(model: tf.keras.Model, ecg_tensor: np.ndarray) -> dict:
    """
    Process the ECG tensor using the ECG2AF model.

    Args:
        model (tf.keras.Model): Loaded ECG2AF model.
        ecg_tensor (np.ndarray): Tensor representation of the ECG data.

    Returns:
        dict: Dictionary containing the model's predictions.
    """
    predictions = model.predict(np.expand_dims(ecg_tensor, axis=0))
    return {
        "AF Risk": predictions[0][0],
        "AF Burden": predictions[1][0],
        "Survival Curve": predictions[2][0].tolist(),
        "ECG Quality": predictions[3][0]
    }

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("ECG2AF Model Web Application")

    # File uploader
    uploaded_file = st.file_uploader("Upload an ECG file (HD5 format)", type=['h5', 'hd5'])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_ecg.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load the model
            model = load_model(MODEL_PATH)

            # Process the ECG file
            ecg_tensor = ecg_as_tensor("temp_ecg.h5")
            results = process_ecg(model, ecg_tensor)

            # Display results
            st.subheader("Prediction Results")
            st.write(f"AF Risk: {results['AF Risk']:.4f}")
            st.write(f"AF Burden: {results['AF Burden']:.4f}")
            st.write(f"ECG Quality: {results['ECG Quality']:.4f}")

            st.subheader("Survival Curve")
            st.line_chart(results['Survival Curve'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            # Clean up the temporary file
            if os.path.exists("temp_ecg.h5"):
                os.remove("temp_ecg.h5")

if __name__ == "__main__":
    main()
