# NLP Preprocessing Project

A comprehensive Natural Language Processing (NLP) preprocessing toolkit built with Streamlit, providing interactive web interfaces for various text processing techniques.

## 📋 Project Overview

This project contains multiple Streamlit applications that demonstrate fundamental NLP preprocessing techniques including tokenization, text cleaning, stemming, lemmatization, and bag of words representation.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit applications
- **Multiple NLP Techniques**: Tokenization, text cleaning, stemming, lemmatization, and bag of words
- **Visual Representations**: Data tables and pie charts for better understanding
- **Real-time Processing**: Instant results with interactive controls
- **Educational Tool**: Perfect for learning NLP concepts

## 📁 Project Structure

```
NlpProject/
├── nlpProject.py          # Main NLP preprocessing app
├── nlpProject2.py         # Extended NLP features
├── nlpProject2Regex.py    # Regex-based text processing
├── nlpProject3.py         # Advanced NLP techniques
├── nlpProject3Regex.py    # Advanced regex operations
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **NLTK** - Natural Language Toolkit
- **spaCy** - Industrial-strength NLP library
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **matplotlib** - Data visualization

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NlpProject
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Install spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Usage

### Running the Main Application

```bash
streamlit run nlpProject.py
```

### Running Other Applications

```bash
streamlit run nlpProject2.py
streamlit run nlpProject2Regex.py
streamlit run nlpProject3.py
streamlit run nlpProject3Regex.py
```

## 🔧 Available NLP Techniques

### 1. Tokenization
- **Sentence Tokenization**: Splits text into sentences
- **Word Tokenization**: Splits text into individual words
- **Character Tokenization**: Splits text into characters

### 2. Text Cleaning
- Converts text to lowercase
- Removes punctuation and numbers
- Eliminates stopwords using spaCy

### 3. Stemming
- **Porter Stemmer**: Conservative stemming approach
- **Lancaster Stemmer**: Aggressive stemming approach
- Side-by-side comparison in tabular format

### 4. Lemmatization
- Uses spaCy for accurate lemmatization
- Displays word, part-of-speech, and lemma
- Contextual word reduction

### 5. Bag of Words (BoW)
- Creates word frequency distribution
- Sortable frequency table
- Visual pie chart of top 10 words

## 💡 Example Usage

1. **Launch the application**
2. **Enter your text** in the text area
3. **Select an NLP technique** from the sidebar
4. **Click "Process Text"** to see results
5. **Explore different techniques** with the same text

## 📊 Sample Input/Output

**Input Text**: "Krishnendu is a student of HIT and he loves his family very much."

**Tokenization Output**:
- Sentences: ['Krishnendu is a student of HIT and he loves his family very much.']
- Words: ['Krishnendu', 'is', 'a', 'student', 'of', 'HIT', 'and', 'he', 'loves', 'his', 'family', 'very', 'much', '.']

**Text Cleaning Output**: "krishnendu student hit loves family much"

## 🔍 File Descriptions

- **nlpProject.py**: Main application with core NLP preprocessing features
- **nlpProject2.py**: Extended version with additional NLP capabilities
- **nlpProject2Regex.py**: Regex-based text processing and pattern matching
- **nlpProject3.py**: Advanced NLP techniques and analysis
- **nlpProject3Regex.py**: Complex regex operations for text manipulation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Support

If you have any questions or issues, please feel free to open an issue in the repository.

## 🎯 Future Enhancements

- [ ] Add more NLP techniques (Named Entity Recognition, Sentiment Analysis)
- [ ] Support for multiple languages
- [ ] Batch processing capabilities
- [ ] Export results to different formats
- [ ] Advanced visualization options

---

**Happy NLP Processing! 🚀**