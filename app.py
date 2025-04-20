from flask import Flask, request, jsonify
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import pickle
import numpy as np
import pandas as pd
import trimesh
import tempfile
import os

#
# Load the model (patient_score.pkl) if available
# ----------------------------
try:
    with open("patient_score.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: patient_score.pkl not found. Please make sure the file exists.")
    exit()

# ----------------------------
# Create Flask app
# ----------------------------
app = Flask(__name__)

# Create an OrthodonticModel class from your provided code
class OrthodonticModel:
    def __init__(self, upper_file, lower_file):
        self.upper_file = upper_file
        self.lower_file = lower_file
        self.upper = self._load_stl(upper_file)
        self.lower = self._load_stl(lower_file)

    def _load_stl(self, file_path):
        mesh = trimesh.load(file_path, file_type='ply', force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Invalid mesh in: {file_path}")
        return mesh.vertices

    def compute_features(self):
        features = {
            "Upper_File": self.upper_file,
            "Lower_File": self.lower_file,
            "Upper_Alignment": self._compute_alignment(self.upper),
            "Lower_Alignment": self._compute_alignment(self.lower),
            "Upper_Marginal_Ridges": self._compute_marginal_ridges(self.upper),
            "Lower_Marginal_Ridges": self._compute_marginal_ridges(self.lower),
            "Upper_Buccolingual_Inclination": self._compute_buccolingual_inclination(self.upper),
            "Lower_Buccolingual_Inclination": self._compute_buccolingual_inclination(self.lower),
            "Occlusal_Relationship": self._compute_occlusal_relationship(),
            "Occlusal_Contacts": self._compute_occlusal_contacts(),
            "Overjet": self._compute_overjet(),
            "Overbite": self._compute_overbite(),
            "Anterior_Open_Bite": self._compute_anterior_open_bite(),
            "Upper_Curve_of_Spee": self._compute_curve_of_spee(self.upper),
            "Lower_Curve_of_Spee": self._compute_curve_of_spee(self.lower),
            "Upper_Crowding": self._compute_crowding(self.upper),
            "Lower_Crowding": self._compute_crowding(self.lower),
            "Upper_Spacing": self._compute_spacing(self.upper),
            "Lower_Spacing": self._compute_spacing(self.lower),
            "Upper_Interproximal_Contacts": self._compute_interproximal_contacts(self.upper),
            "Lower_Interproximal_Contacts": self._compute_interproximal_contacts(self.lower),
            "Upper_Root_Angulation": self._compute_root_angulation(self.upper),
            "Lower_Root_Angulation": self._compute_root_angulation(self.lower),
        }
        return features

    def _compute_alignment(self, points):
        pca = PCA(n_components=1)
        pca.fit(points)
        line = pca.components_[0]
        centroid = np.mean(points, axis=0)
        projection = np.dot(points - centroid, line[:, np.newaxis])[:, 0]
        reconstructed = np.outer(projection, line) + centroid
        return np.mean(np.linalg.norm(points - reconstructed, axis=1))

    def _compute_marginal_ridges(self, points):
        z = points[:, 2]
        return np.percentile(z, 95) - np.percentile(z, 5)

    def _compute_buccolingual_inclination(self, points):
        xz = points[:, [0, 2]]
        pca = PCA(n_components=1)
        pca.fit(xz)
        angle = np.arctan2(pca.components_[0][0], pca.components_[0][1])
        return np.degrees(angle)

    def _compute_occlusal_relationship(self):
        return np.mean(self.upper[:, 2]) - np.mean(self.lower[:, 2])

    def _compute_occlusal_contacts(self, threshold=0.5):
        tree = KDTree(self.lower)
        return sum(tree.query(point)[0] < threshold for point in self.upper)

    def _compute_overjet(self):
        upper_front = self.upper[np.argsort(self.upper[:, 1])[-20:]]
        lower_front = self.lower[np.argsort(self.lower[:, 1])[-20:]]
        return np.mean(upper_front[:, 0]) - np.mean(lower_front[:, 0])

    def _compute_overbite(self):
        upper_incisors = self.upper[np.argsort(self.upper[:, 1])[-20:]]
        lower_incisors = self.lower[np.argsort(self.lower[:, 1])[-20:]]
        return np.mean(lower_incisors[:, 2]) - np.mean(upper_incisors[:, 2])

    def _compute_anterior_open_bite(self):
        bite = self._compute_overbite()
        return abs(bite) if bite < 0 else 0

    def _compute_curve_of_spee(self, points):
        y_sorted = points[np.argsort(points[:, 1])]
        z_curve = y_sorted[:, 2]
        return np.max(z_curve) - np.min(z_curve)

    def _compute_crowding(self, points):
        gaps = np.diff(np.sort(points[:, 0]))
        return np.sum(gaps < 0.6) * 0.5

    def _compute_spacing(self, points):
        gaps = np.diff(np.sort(points[:, 0]))
        return np.sum(gaps > 1.5) * 1.0

    def _compute_interproximal_contacts(self, points):
        x = np.sort(points[:, 0])
        return np.mean(np.diff(x))

    def _compute_root_angulation(self, points):
        return np.std(points[:, 2])

# --------------------------
# File Upload and Prediction
# --------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the uploaded files from the request
        upper_file = request.files.get('upper_file')
        lower_file = request.files.get('lower_file')

        if not upper_file or not lower_file:
            return jsonify({"error": "Both upper and lower STL files are required."}), 400

        # Save the uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_upper, tempfile.NamedTemporaryFile(delete=False) as temp_lower:
            upper_path = temp_upper.name
            lower_path = temp_lower.name

            upper_file.save(upper_path)
            lower_file.save(lower_path)

            # Initialize the orthodontic model with the uploaded files
            model = OrthodonticModel(upper_path, lower_path)
            features = model.compute_features()

            # Return the computed features as a response
            return jsonify(features)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
