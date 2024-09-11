import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Affiche une introduction à la première partie du traitement.
print("######  Partie 1 : Préparation des données  ######\n")

# Chargement des données depuis un fichier CSV.
data = pd.read_csv('synthetic.csv')

# Séparation des données en features et en target (la colonne 'Class' est la cible).
X = data.drop('Class', axis=1)
y = data['Class']

# Division des données en ensembles d'entraînement et de test avec 20% des données pour le test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Affichage des tailles des ensembles d'entraînement et de test pour vérification.
print("Taille des données d'entraînement:", X_train.shape)
print("Taille des données de test:", X_test.shape)

# Affichage des informations statistiques des données.
print("\nQuestion 1 : Nombre d'attributs:", len(data.columns) - 1, "\n")
print("Question 2 : Classes différentes:", data['Class'].nunique(), "\n")
print("Question 3 : Instances par classe:", data['Class'].value_counts(), "\n")

print("Question 4 : Linéarité?\n")

# Création d'un scatter plot pour visualiser la séparabilité des classes.
plt.scatter(data['Attr_A'], data['Attr_B'], c=data['Class'])
plt.xlabel('Attr_A')
plt.ylabel('Attr_B')
plt.title('Visualisation de la séparabilité des classes selon Attr_A et Attr_B')
plt.colorbar()
plt.show()

print("\n######  Partie 2 : Mise en oeuvre de modeles  ######\n")
print("###  2.1 : Arbre de décisions  ### \n")

# Calcul des quartiles pour utiliser comme seuils potentiels dans l'arbre de décision.
quartiles = X_train.quantile([0.25, 0.5, 0.75]).to_dict()


# Fonction pour calculer l'entropie d'un ensemble de labels.
def calculate_entropy(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy


print("#----# TEST de la fonction de calculate_entropy")
entropy_value = calculate_entropy(data['Class'].values)
print(f"Entropie de la colonne 'Class':{entropy_value} \n#----#")


# Fonction pour calculer le gain d'information.
def information_gain(X, y, feature, threshold):
    left_mask = X[feature] <= threshold
    right_mask = ~left_mask
    left_child = y[left_mask]
    right_child = y[right_mask]
    entropy_parent = calculate_entropy(y)
    entropy_left = calculate_entropy(left_child)
    entropy_right = calculate_entropy(right_child)
    size_parent = len(y)
    size_left = len(left_child)
    size_right = len(right_child)
    gain = entropy_parent - (size_left / size_parent * entropy_left + size_right / size_parent * entropy_right)
    return gain


print("#----# TEST de la fonction information_gain")
feature = 'Attr_A'
threshold = data[feature].median()
gain_value = information_gain(data, data['Class'], feature, threshold)
print(f"Gain d'information pour {feature} avec un seuil de {threshold:.4f}: {gain_value:.4f}\n#----#")


# Fonction pour trouver le meilleur split.
def best_split(X, y):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    for feature in X.columns:
        for quartile in quartiles[feature].values():
            gain = information_gain(X, y, feature, quartile)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = quartile
    return best_feature, best_threshold, best_gain


print("#----# TEST de la fonction best_split")
best_feature, best_threshold, best_gain = best_split(data.drop('Class', axis=1), data['Class'])
print(f"Meilleur split: Feature = {best_feature}, Threshold = {best_threshold:.4f}, Gain = {best_gain:.4f}\n#----#")


# Classe représentant l'arbre de décision.
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) <= 1:
            return np.bincount(y).argmax()
        best_feature, best_threshold, _ = best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()
        node = {'feature': best_feature, 'threshold': best_threshold, 'left': None, 'right': None}
        left_mask = X[best_feature] <= best_threshold
        right_mask = ~left_mask
        node['left'] = self.fit(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self.fit(X[right_mask], y[right_mask], depth + 1)
        self.tree = node
        return node

    def predict(self, X):
        def _predict_one(row, node):
            if isinstance(node, dict):
                if row[node['feature']] <= node['threshold']:
                    return _predict_one(row, node['left'])
                else:
                    return _predict_one(row, node['right'])
            else:
                return node

        return X.apply(lambda row: _predict_one(row, self.tree), axis=1)


# Fonction pour calculer la précision d'un modèle.
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Test de différentes profondeurs pour l'arbre de décision.
depths = range(3, 9)
results = []
for depth in depths:
    tree_model = DecisionTree(max_depth=depth)
    tree_model.fit(X_train, y_train)
    predictions = tree_model.predict(X_test)
    acc = accuracy(y_test, predictions)
    results.append((depth, acc))
    print(f"Profondeur: {depth}: Précision: {acc:.4f}")

# Affichage graphique des résultats de précision.
plt.figure(figsize=(10, 5))
depths = [result[0] for result in results]
accuracies = [result[1] for result in results]
plt.plot(depths, accuracies, marker='o', linestyle='-', color='b')
plt.title('Précision de l\'arbre de décision en fonction de la profondeur')
plt.xlabel('Profondeur de l\'arbre')
plt.ylabel('Précision')
plt.xticks(depths)
plt.grid(True)
plt.show()

print("\n###  2.2 : Réseaux de neurones  ### \n")


# Classe représentant un réseau de neurones.
class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', reg_lambda=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) / np.sqrt(layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward_pass(self, X):
        activation = X
        activations = [X]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activation, w) + b
            if self.activation == 'relu' and w is not self.weights[-1]:
                activation = np.maximum(0, z)
            elif self.activation == 'tanh' and w is not self.weights[-1]:
                activation = np.tanh(z)
            else:
                c = np.max(z, axis=1, keepdims=True)
                exp_scores = np.exp(z - c)
                activation = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            zs.append(z)
            activations.append(activation)
        return zs, activations

    def compute_error(self, Y, Y_hat):
        correct_logprobs = -np.log(Y_hat[range(len(Y_hat)), Y.argmax(axis=1)] + 1e-9)
        data_loss = np.sum(correct_logprobs) / len(Y_hat)
        return data_loss

    def backward_pass(self, X, Y, zs, activations, learning_rate):
        delta = activations[-1] - Y
        deltas = [delta]

        for l in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[l].T)
            if self.activation == 'relu':
                delta[zs[l - 1] <= 0] = 0
            elif self.activation == 'tanh':
                delta = delta * (1 - np.power(activations[l], 2))
            deltas.append(delta)

        deltas.reverse()

        for l in range(len(self.weights)):
            dW = np.dot(activations[l].T, deltas[l])
            db = np.sum(deltas[l], axis=0, keepdims=True)
            self.weights[l] -= learning_rate * (dW + self.reg_lambda * self.weights[l])
            self.biases[l] -= learning_rate * db

    def train(self, X_train, y_train_one_hot, X_val, y_val_one_hot, X_test, y_test_one_hot, epochs, learning_rate,
              batch_size, patience=4):
        n = len(X_train)
        best_val_error = float('inf')
        epochs_no_improve = 0
        val_errors = []
        test_errors = []

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train_one_hot[perm]

            for i in range(0, n, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                zs, activations = self.forward_pass(X_batch)
                self.backward_pass(X_batch, y_batch, zs, activations, learning_rate)

            val_predictions = self.forward_pass(X_val)[1][-1]
            val_error = self.compute_error(y_val_one_hot, val_predictions)
            val_errors.append(val_error)

            test_predictions = self.forward_pass(X_test)[1][-1]
            test_error = self.compute_error(y_test_one_hot, test_predictions)
            test_errors.append(test_error)

            print(f"Epoch {epoch}, Validation Error: {val_error}, Test Error: {test_error}")

            if val_error < best_val_error:
                best_val_error = val_error
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping!")
                    break

        return val_errors, test_errors


print("#----# TEST de la fonction forward_pass")
testnn = NeuralNetwork(layer_sizes=[data.shape[1] - 1, 10, 5, 2], activation='relu')

X_sample_test = np.random.randn(1, data.shape[1] - 1)
Y_sample_test = np.array([[0, 1]])

zs_test, activations_test = testnn.forward_pass(X_sample_test)
print(f"Activations après la passe avant: {activations_test[-1]}\n#----#")

print("#----# TEST de la fonction backward_pass")
testnn.backward_pass(X_sample_test, Y_sample_test, zs_test, activations_test, learning_rate=0.01)
print(f"Poids mis à jour (exemple pour la première couche):{testnn.weights[0]}\n#----#\n")

# Normalisation des données et one-hot encoding pour les labels.
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y_one_hot = pd.get_dummies(y)
X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split(X, y_one_hot, test_size=0.20, random_state=42)
X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X_train, y_train_one_hot, test_size=0.15,
                                                                  random_state=42)

# Définition des architectures pour les tests.
architectures = {
    'tanh': [(10, 8, 6), (10, 8, 4), (6, 4)],
    'relu': [(10, 8, 6), (10, 8, 4), (6, 4)]
}

results = {}
for activation in architectures:
    results[activation] = {}
    for arch in architectures[activation]:
        print(f"{arch} avec {activation}:\n")
        nn = NeuralNetwork(layer_sizes=[X_train.shape[1]] + list(arch) + [y_train_one_hot.shape[1]],
                           activation=activation)
        val_errors, test_errors = nn.train(X_train.values, y_train_one_hot.values, X_val.values, y_val_one_hot.values,
                                           X_test.values, y_test_one_hot.values, 100, 0.01, 4, patience=4)

        plt.figure()
        plt.plot(val_errors, label='Validation Error')
        plt.plot(test_errors, label='Test Error')
        plt.title(f'Error Progression: {activation.upper()} {arch}')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.show()

        results[activation][arch] = {'val_error': val_errors[-1], 'test_error': test_errors[-1]}
        print(
            f"Final Validation Error for {arch} avec {activation}: {val_errors[-1]}, Final Test Error: {test_errors[-1]},\n")
        print("\n=======================\n")

print("\n######  Partie 3 : Analyse des modeles  ######\n")


# Fonction pour calculer les vrais positifs, faux positifs, et faux négatifs pour une classe spécifique.
def calculate_tp_fp_fn(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    return tp, fp, fn


def calculate_metrics(y_true, y_pred):
    labels = np.unique(y_true)
    class_accuracy = []  # Pour stocker l'accuracy de chaque classe
    precision = []
    recall = []
    f1 = []

    for label in labels:
        tp, fp, fn = calculate_tp_fp_fn(y_true, y_pred, label)
        class_acc = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0  # Calcule l'accuracy pour chaque classe
        class_accuracy.append(class_acc)
        if tp + fp == 0:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))
        if tp + fn == 0:
            recall.append(0)
        else:
            recall.append(tp / (tp + fn))
        if precision[-1] + recall[-1] == 0:
            f1.append(0)
        else:
            f1.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))

    return class_accuracy, precision, recall, f1


def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            matrix[i, j] = np.sum((y_true == label1) & (y_pred == label2))
    return matrix


def format_metrics_and_confusion(metrics, conf_matrix, model_name):
    class_accuracy, precision, recall, f1 = metrics
    num_classes = len(class_accuracy)  # Supposant que toutes les métriques ont le même nombre de classes
    columns = [f'C{i}' for i in range(num_classes)]

    # Création d'un DataFrame pour chaque métrique
    data = {
        'Metrics': ['Class Accuracy', 'Precision', 'Recall', 'F1-score'],
    }
    data.update({col: [] for col in columns})  # Ajout des colonnes pour chaque classe

    # Remplissage des colonnes pour chaque métrique
    for metric, values in zip(data['Metrics'], [class_accuracy, precision, recall, f1]):
        for col, val in zip(columns, values):
            data[col].append(val)

    report_df = pd.DataFrame(data)
    print("Table 1 - Classification report for", model_name)
    print(report_df.to_string(index=False))
    print("\n")

    conf_df = pd.DataFrame(conf_matrix, index=columns, columns=columns)
    conf_df.index.name = 'True label'
    conf_df.columns.name = 'Predicted label'
    print("Table 2 - Confusion Matrix for model", model_name)
    print(conf_df)
    print("\n ================================ \n")


# Conversion des probabilités en classes pour les modèles de réseaux de neurones.
def convert_probabilities_to_classes(df):
    return np.argmax(df.values, axis=1)


# Chargement des données de test et des prédictions de différents modèles.
y_test = pd.read_csv('Predictions/y_test.csv')
y_pred_DT4 = pd.read_csv('Predictions/y_pred_DT4.csv')
y_pred_DT5 = pd.read_csv('Predictions/y_pred_DT5.csv')
y_pred_DT6 = pd.read_csv('Predictions/y_pred_DT6.csv')
y_pred_NN_relu_6_4 = pd.read_csv('Predictions/y_pred_NN_relu_6-4.csv')
y_pred_NN_relu_10_8_4 = pd.read_csv('Predictions/y_pred_NN_relu_10-8-4.csv')
y_pred_NN_relu_10_8_6 = pd.read_csv('Predictions/y_pred_NN_relu_10-8-6.csv')
y_pred_NN_tanh_6_4 = pd.read_csv('Predictions/y_pred_NN_tanh_6-4.csv')
y_pred_NN_tanh_10_8_4 = pd.read_csv('Predictions/y_pred_NN_tanh_10-8-4.csv')
y_pred_NN_tanh_10_8_6 = pd.read_csv('Predictions/y_pred_NN_tanh_10-8-6.csv')

# Renommage des colonnes pour une uniformité des données.
y_test.rename(columns={y_test.columns[0]: 'true_label'}, inplace=True)
y_pred_DT4.rename(columns={y_pred_DT4.columns[0]: 'predicted'}, inplace=True)
y_pred_DT5.rename(columns={y_pred_DT5.columns[0]: 'predicted'}, inplace=True)
y_pred_DT6.rename(columns={y_pred_DT6.columns[0]: 'predicted'}, inplace=True)

# Calcul et affichage des métriques et des matrices de confusion pour chaque modèle.
predictions = {
    "DT4": y_pred_DT4['predicted'],
    "DT5": y_pred_DT5['predicted'],
    "DT6": y_pred_DT6['predicted'],
    "NN_relu_6_4": convert_probabilities_to_classes(y_pred_NN_relu_6_4),
    "NN_relu_10_8_4": convert_probabilities_to_classes(y_pred_NN_relu_10_8_4),
    "NN_relu_10_8_6": convert_probabilities_to_classes(y_pred_NN_relu_10_8_6),
    "NN_tanh_6_4": convert_probabilities_to_classes(y_pred_NN_tanh_6_4),
    "NN_tanh_10_8_4": convert_probabilities_to_classes(y_pred_NN_tanh_10_8_4),
    "NN_tanh_10_8_6": convert_probabilities_to_classes(y_pred_NN_tanh_10_8_6)
}

for model_name, y_pred in predictions.items():
    metrics = calculate_metrics(y_test['true_label'], y_pred)
    confusion = confusion_matrix(y_test['true_label'], y_pred)
    format_metrics_and_confusion(metrics, confusion, model_name)
