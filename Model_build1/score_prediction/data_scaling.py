from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    """
    Scales the training and testing data using StandardScaler.
    
    Parameters:
    X_train: pd.DataFrame or np.array
        Training data features
    X_test: pd.DataFrame or np.array
        Testing data features
    
    Returns:
    X_train_scaled, X_test_scaled
    """
    scaler = StandardScaler()
    
    # Fit the scaler on training data and transform both train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
