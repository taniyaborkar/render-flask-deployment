from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import trimesh
import tempfile

# Load ML model
with open("ortho2.pkl", "rb") as f:
    ml_model = pickle.load(f)

app = Flask(__name__)

def load_ply(file_path):
    mesh = trimesh.load(file_path, file_type='ply', force='mesh')
    return mesh.vertices if isinstance(mesh, trimesh.Trimesh) else None

def compute_features(upper_points, lower_points):
    def alignment(points):
        pca = PCA(n_components=1)
        pca.fit(points)
        line = pca.components_[0]
        centroid = np.mean(points, axis=0)
        projection = np.dot(points - centroid, line[:, np.newaxis])[:, 0]
        reconstructed = np.outer(projection, line) + centroid
        return np.mean(np.linalg.norm(points - reconstructed, axis=1))

    def marginal_ridges(points):
        z = points[:, 2]
        return np.percentile(z, 95) - np.percentile(z, 5)

    def buccolingual_inclination(points):
        xz = points[:, [0, 2]]
        pca = PCA(n_components=1)
        pca.fit(xz)
        angle = np.arctan2(pca.components_[0][0], pca.components_[0][1])
        return np.degrees(angle)

    def occlusal_relationship():
        return np.mean(upper_points[:, 2]) - np.mean(lower_points[:, 2])

    def occlusal_contacts(threshold=0.5):
        tree = KDTree(lower_points)
        return sum(tree.query(p)[0] < threshold for p in upper_points)

    def overjet():
        upper_front = upper_points[np.argsort(upper_points[:, 1])[-20:]]
        lower_front = lower_points[np.argsort(lower_points[:, 1])[-20:]]
        return np.mean(upper_front[:, 0]) - np.mean(lower_front[:, 0])

    def interproximal_contacts(points):
        x = np.sort(points[:, 0])
        return np.mean(np.diff(x))

    def root_angulation(points):
        return np.std(points[:, 2])

    return {
        "Alignment": (alignment(upper_points) + alignment(lower_points)) / 2,
        "Marginal_Ridges": (marginal_ridges(upper_points) + marginal_ridges(lower_points)) / 2,
        "Buccolingual_Inclination": (buccolingual_inclination(upper_points) + buccolingual_inclination(lower_points)) / 2,
        "Occlusal_Relationships": occlusal_relationship(),
        "Occlusal_Contacts": occlusal_contacts(),
        "Overjet": overjet(),
        "Interproximal_Contacts": (interproximal_contacts(upper_points) + interproximal_contacts(lower_points)) / 2,
        "Root_Angulation": (root_angulation(upper_points) + root_angulation(lower_points)) / 2,
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        upper_file = request.files.get("upper_file")
        lower_file = request.files.get("lower_file")

        if not upper_file or not lower_file:
            return jsonify({"error": "Both upper and lower PLY files are required."}), 400

        with tempfile.NamedTemporaryFile(delete=False) as temp_upper, tempfile.NamedTemporaryFile(delete=False) as temp_lower:
            upper_file.save(temp_upper.name)
            lower_file.save(temp_lower.name)

            upper_points = load_ply(temp_upper.name)
            lower_points = load_ply(temp_lower.name)

            if upper_points is None or lower_points is None:
                return jsonify({"error": "Invalid PLY files"}), 400

            features = compute_features(upper_points, lower_points)

            # Prepare for prediction
            input_data = np.array([list(features.values())])
            prediction = ml_model.predict(input_data)[0]

            return jsonify({
                "features": features,
                "ABO_Compliant": bool(prediction)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Running"})

if __name__ == "__main__":
    app.run(debug=True)
