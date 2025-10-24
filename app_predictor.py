import os
from flask import Flask, render_template, request
import numpy as np
from linear_regression_analytical import feature_cols, numeric_idx, X_mean, X_std, theta, district_cols


template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=template_path)

district_names = [d.replace("district_", "") for d in district_cols]

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        surface = float(request.form["surface"])
        rooms = int(request.form["rooms"])
        district = request.form["district"]

        x_test = {c: 0 for c in feature_cols}
        x_test["surface_num"] = surface
        x_test["no_of_rooms"] = rooms
        x_test["bias"] = 1

        for col in district_cols:
            if district in col:
                x_test[col] = 1

        for d in district_cols:
            x_test[f"{d}_x_surface"] = x_test[d] * surface
            x_test[f"{d}_x_rooms"] = x_test[d] * rooms

        x_vec = np.array([x_test[c] for c in feature_cols]).reshape(1, -1)
        x_vec[:, numeric_idx] = (x_vec[:, numeric_idx] - X_mean) / X_std

        predicted_price = (x_vec @ theta).item()

    return render_template(
        "index.html",
        price=predicted_price,
        districts=district_names,
        surface=surface,
        rooms=rooms,
        district=district
    )


if __name__ == "__main__":
    app.run(debug=True)
