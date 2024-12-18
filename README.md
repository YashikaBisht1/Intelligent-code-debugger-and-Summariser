# Intelligent-code-debugger-and-Summariser

This repository contains the **Intelligent Code Debugger and Summarizer**, a Streamlit-based web application designed to help developers analyze Python code. It provides functionality for detecting syntax errors, unused variables, cyclic dependencies, generating code fixes, and summarizing code using AI models.

## Features

- **Syntax Analysis**: Detects syntax errors in uploaded Python code using Python's Abstract Syntax Tree (AST).
- **Unused Variable Detection**: Identifies variables declared but never used in the code.
- **Cyclic Dependency Detection**: Uses graph traversal techniques to identify cyclic dependencies in code.
- **Bug Fix Suggestions**: Leverages AI models (e.g., CodeT5) to generate fixes for buggy Python code.
- **Code Summarization**: Uses Google GenAI to summarize and analyze code functionality.
- **Debugging History**: Maintains a history of debugging sessions to revisit buggy and fixed code.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/intelligent-code-debugger.git
   cd intelligent-code-debugger
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   - Create a `.env` file in the root directory.
   - Add your Google API key:
     ```env
     gemini_api=YOUR_GOOGLE_API_KEY
     ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a Python file (`.py`) through the web interface.
2. The app will analyze the code and display:
   - Detected syntax errors.
   - Unused variables.
3. Generate bug fixes using AI models.
4. Summarize the code for a better understanding of its functionality.
5. View the history of debugging sessions.

## File Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # List of dependencies
├── .env                 # Environment variables (not included in the repo)
```

## Models Used

- **CodeT5**: A pre-trained model for code understanding and generation.
- **Google GenAI**: For summarizing and analyzing code functionality.

## Examples

### Uploaded Code

```python
x = 10
y = 20
z = x + y
print(a)  # Undefined variable
```

### Detected Issues

- **Syntax Error**: `NameError: name 'a' is not defined`
- **Unused Variables**: `y`

### Suggested Fix

```python
x = 10
z = x + 10
print(z)
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or suggestions, feel free to reach out:

- Email: [bishty2005@gmail.com](mailto\:bishty2005@gmail.com)
- LinkedIn: [Yashika Bisht](https://linkedin.com/in/yashika-bisht-20050a24a)

