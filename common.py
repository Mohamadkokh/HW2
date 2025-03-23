# common.py
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression

class MyPreprocessing(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.num_features = X.select_dtypes(["int64", "float64"]).columns
        self.cat_features = X.select_dtypes(["object"]).columns

        self.num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]
        )
        self.cat_transformer = OneHotEncoder()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.num_transformer, self.num_features),
                ("cat", self.cat_transformer, self.cat_features),
            ]
        )

        self.preprocessor.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.preprocessor.transform(X)

class MyClassifier(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegression(
            solver="saga", max_iter=500, penalty="l2", random_state=42
        )

    def fit(self, X, y=None):
        self.clf.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        check_is_fitted(self)
        return self.clf.predict(X)
