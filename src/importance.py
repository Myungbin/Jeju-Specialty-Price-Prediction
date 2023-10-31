import seaborn as sns
from matplotlib import pyplot as plt


def xgb_importance(model):
    feature_importance = model.feature_importances_
    sns.barplot(x=feature_importance, y=x_train.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Variable Importance Plot')
    plt.show()


def cat_importance(model):
    feature_importance = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(x_train.columns, feature_importance, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Variable Importance Plot')
    plt.show()
