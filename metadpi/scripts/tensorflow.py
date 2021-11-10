import tensorflow_decision_forests as tfdf

def tfrf(test_frame, train_frame, annotatted_col):
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_frame, label =annotatted_col )
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_frame,label = annotatted_col)
    model_1 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
    model_1.fit(x=train_ds)
    prediction = model_1.predict(test_ds)
    test_frame["randomforesttf"] = prediction
    print(test_frame)
    return