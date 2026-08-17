import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from wordcloud import WordCloud
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
import numpy as np

def evaluate_model(y_true, y_pred, classes, assets_path='assets'):
    """
    Evaluate model performance and save confusion matrix plot.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the plot
    plot_path = f"{assets_path}/confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\nConfusion matrix saved to {plot_path}")
    plt.close()
    
    return accuracy, precision, recall, f1

def error_analysis(test_texts, y_test, predictions, probabilities):
    """
    Perform and display error analysis.
    """
    print("\n=== Error Analysis ===")
    errors = []
    
    for i, (true_label, pred_label, text, probs) in enumerate(
        zip(y_test, predictions, test_texts, probabilities)
    ):
        if true_label != pred_label:
            confidence = max(probs.values())
            errors.append({
                'index': i,
                'text': text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'probabilities': probs
            })
    
    if not errors:
        print("No errors found on the test set. Perfect classification!")
        return

    print(f"Total errors: {len(errors)} out of {len(y_test)} samples.")
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("\nTop 5 most confident misclassifications:")
    for i, error in enumerate(errors[:5]):
        print(f"\n{i+1}. Text: '{error['text']}'")
        print(f"   True: {error['true_label']}, Predicted: {error['predicted_label']}")
        print(f"   Confidence: {error['confidence']:.4f}")
        print(f"   Probabilities: {error['probabilities']}")

def plot_classification_report(y_true, y_pred, classes, assets_path='assets'):
    """
    Generate and save a heatmap for the classification report.
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Classification Report')
    
    plot_path = f"{assets_path}/classification_report.png"
    plt.savefig(plot_path)
    print(f"Classification report heatmap saved to {plot_path}")
    plt.close()

def plot_word_clouds(feature_importance, assets_path='assets'):
    """
    Generate and save word clouds for positive and negative features.
    """
    if not feature_importance:
        print("Cannot generate word clouds due to missing feature importance data.")
        return

    # Separate features into positive and negative based on their score
    pos_features = {k: v for k, v in feature_importance.items() if v > 0}
    neg_features = {k: abs(v) for k, v in feature_importance.items() if v < 0}

    if not pos_features or not neg_features:
        print("Not enough feature data to generate word clouds.")
        return

    # Positive Word Cloud
    plt.figure(figsize=(10, 7))
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pos_features)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Positive Words')
    pos_plot_path = f"{assets_path}/wordcloud_positive.png"
    plt.savefig(pos_plot_path)
    print(f"Positive word cloud saved to {pos_plot_path}")
    plt.close()

    # Negative Word Cloud
    plt.figure(figsize=(10, 7))
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate_from_frequencies(neg_features)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Negative Words')
    neg_plot_path = f"{assets_path}/wordcloud_negative.png"
    plt.savefig(neg_plot_path)
    print(f"Negative word cloud saved to {neg_plot_path}")
    plt.close()

def plot_feature_importance(feature_importance, top_n=15, assets_path='assets'):
    """
    Plot and save the most important features for classification.
    """
    if not feature_importance:
        print("Cannot plot feature importance due to missing data.")
        return

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
    
    top_neg = sorted_features[:top_n]
    top_pos = sorted_features[-top_n:]
    
    # Combine for plotting
    top_features = top_pos + top_neg
    
    words = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if s > 0 else 'red' for s in scores]
    plt.barh(np.arange(len(words)), scores, color=colors)
    plt.yticks(np.arange(len(words)), words)
    plt.xlabel('Feature Importance (Log Probability Difference)')
    plt.title(f'Top {top_n} Most Important Features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plot_path = f"{assets_path}/feature_importance.png"
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")
    plt.close()

def plot_roc_curve(y_true, y_prob, classes, assets_path='assets'):
    """
    Plot the ROC curve and calculate AUC.
    """
    # Assuming 'positive' is the positive class
    pos_class_label = 'positive'
    y_true_binary = [1 if label == pos_class_label else 0 for label in y_true]
    y_prob_scores = [p[pos_class_label] for p in y_prob]
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plot_path = f"{assets_path}/roc_curve.png"
    plt.savefig(plot_path)
    print(f"ROC curve plot saved to {plot_path}")
    plt.close()
