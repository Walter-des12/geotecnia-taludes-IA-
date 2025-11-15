from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from lightgbm import LGBMRegressor
import joblib

def entrenar_modelo(df, ruta_salida):

    X = df.drop(columns=["Factor_de_seguridad_FS"])
    y = df["Factor_de_seguridad_FS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        lazy = LazyRegressor(verbose=0, ignore_warnings=True)
        modelos, preds = lazy.fit(X_train, X_test, y_train, y_test)
        best = modelos["Adjusted R-Squared"].idxmax()
        model = lazy.models[best]
        model.fit(X_train, y_train)
        print("Modelo elegido automáticamente:", best)

    except:
        print("LazyPredict falló, usando LightGBM")
        model = LGBMRegressor()
        model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "feature_names": X.columns.tolist(),
            "X_test": X_test,
            "y_test": y_test
        },
        ruta_salida
    )

    print("\n✓ ENTRENAMIENTO COMPLETADO")
    print("✓ Modelo guardado en:", ruta_salida)

    return model


if __name__ == "__main__":
    from load_data import cargar_dataset
    df = cargar_dataset("data/slope_stability_dataset.csv")
    entrenar_modelo(df, "models/best_fs_model.joblib")
