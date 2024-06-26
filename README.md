1. Importing Libraries:

pandas (pd):Used for data manipulation (reading CSV, creating DataFrames)
numpy (np): Used for numerical computations (not explicitly used here)
sklearn.preprocessing.LabelEncoder: Converts text labels ("spam", "ham") to numbers for machine learning models
matplotlib.pyplot (plt): Used for data visualization (pie charts, histograms)
nltk: Used for Natural Language Processing (text tokenization, sentence segmentation)
seaborn (sns): Advanced data visualization library built on top of matplotlib
wordcloud: Creates word clouds to visualize word frequency
collections.Counter: Counts occurrences of elements in a collection
sklearn.feature_extraction.text: Feature extraction from text data (CountVectorizer, TfidfVectorizer)
sklearn.model_selection: Train-test split for model evaluation
sklearn.naive_bayes: Implements Naive Bayes classification algorithms
sklearn.metrics: Provides metrics for model evaluation (accuracy, confusion matrix, precision)
pickle: Saves Python objects (models and vectorizer) for later use

2. Data Loading and Preprocessing:

Reads the CSV file 'spam.csv' using pd.read_csv with encoding set to 'latin' (assuming the data uses Latin-1 encoding).
Drops unnecessary columns named 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4' (likely leftover from data import).
Renames columns 'v1' to 'target' (spam label) and 'v2' to 'text' (email content).
Uses LabelEncoder to convert the text labels in 'target' to numerical values (0 for ham, 1 for spam).

3. Exploratory Data Analysis (EDA):

Checks for missing values using df.isnull().sum().
Identifies and removes duplicate entries using df.drop_duplicates().
Analyzes the distribution of spam/ham labels using df['target'].value_counts().
Creates a pie chart to visualize the proportions of spam and ham emails with plt.pie.

4. Text Preprocessing:

Defines a function transform_text to perform the following:
Converts text to lowercase.
Tokenizes the text using nltk.word_tokenize.
Filters out non-alphanumeric characters and punctuation.
Removes stopwords (common words like "the", "a") using nltk.corpus.stopwords.words("English").
Applies stemming (reducing words to their root form) using nltk.stem.porter.PorterStemmer.
Joins the processed tokens back into a string.
Applies the transform_text function to the 'text' column and stores the result in a new column named 'transformed_text'.

5. Feature Engineering:

Creates new features based on the text data:
num_characters: Length of the email text.
num_words: Number of words in the email text (after tokenization).
num_sentences: Number of sentences in the email text (using nltk.sent_tokenize).
Analyzes the distribution of these features for spam and ham emails using sns.histplot.

6. Word Clouds:

Generates word clouds to visualize frequently used words in spam and ham emails using wordcloud.WordCloud.
Creates separate word clouds for spam and ham emails using df[df['target']==1]... and df[df['target']==0]... to filter data.

7. Feature Extraction:

Creates two feature extraction techniques:
TfidfVectorizer: Creates a TF-IDF (Term Frequency-Inverse Document Frequency) matrix, which considers both word frequency and document importance.
CountVectorizer: Creates a simple word count matrix, where each element represents the count of a word in a document.
Fits the vectorizer (tfid) to the transformed text data (df['transformed_text']). This creates a vocabulary and learns word weights.

8. Model Training and Evaluation:

Splits the data into training and testing sets using sklearn.model_selection.train_test_split.
Trains three Naive Bayes classifiers:
BernoulliNB: Suitable for binary features (presence/absence of words).
GaussianNB: Assumes Gaussian distribution of features (not ideal for text data).
`MultinomialNB
