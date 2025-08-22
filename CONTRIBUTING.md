# Contributing to Fall Detection System

We welcome contributions to the Fall Detection System for Elderly Care! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Conda or virtualenv
- Basic understanding of machine learning concepts
- Familiarity with Flask, Socket.IO, and TensorFlow

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/klasifikasi-lansia-jatuh.git
   cd klasifikasi-lansia-jatuh
   ```

## Development Setup

### Environment Setup

1. Create a conda environment:
   ```bash
   conda create -n comvis python=3.11
   conda activate comvis
   ```

2. Install dependencies:
   ```bash
   cd laptop
   pip install -r requirements.txt
   ```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Firebase Setup

1. Create a Firebase project
2. Generate service account credentials
3. Place the JSON file in the `laptop/` directory
4. Update configuration accordingly

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Add new functionality
- **Documentation**: Improve or add documentation
- **Performance improvements**: Optimize existing code
- **Testing**: Add or improve tests
- **UI/UX improvements**: Enhance the web dashboard

### Before You Start

1. Check existing issues to avoid duplication
2. Create an issue to discuss major changes
3. Ensure your development environment is set up correctly
4. Read through the codebase to understand the architecture

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style and conventions
- Add comments for complex logic
- Update documentation if necessary

### 3. Test Your Changes

- Test the laptop server functionality
- Test the Raspberry Pi client (if applicable)
- Verify the web dashboard works correctly
- Check Firebase integration
- Ensure no existing functionality is broken

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of your changes"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots (if UI changes)
- Testing instructions

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, dependencies
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error messages and stack traces
- **Screenshots**: If applicable

### Feature Requests

For feature requests, please provide:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Any other relevant information

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Write docstrings for functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### File Organization

- Keep related functionality together
- Use clear, descriptive file names
- Maintain consistent directory structure
- Update imports when moving files

### Documentation

- Update README.md for significant changes
- Add inline comments for complex logic
- Update API documentation
- Include examples in docstrings

## Testing

### Manual Testing

1. **Server Testing**:
   - Start the laptop server
   - Verify web dashboard loads
   - Test real-time communication
   - Check Firebase integration

2. **Client Testing**:
   - Test Raspberry Pi client connection
   - Verify video capture and processing
   - Check data transmission

3. **Integration Testing**:
   - Test end-to-end fall detection
   - Verify logging functionality
   - Check alert system

### Automated Testing

We encourage adding automated tests:

- Unit tests for individual functions
- Integration tests for component interaction
- End-to-end tests for complete workflows

## Development Tips

### Debugging

- Use debug mode for development: `python server.py --debug`
- Check console logs and CSV files
- Monitor Firebase console for data flow
- Use browser developer tools for frontend issues

### Performance

- Profile code for performance bottlenecks
- Optimize model inference time
- Monitor memory usage
- Consider caching strategies

### Security

- Never commit sensitive information
- Use environment variables for configuration
- Follow security best practices
- Validate all user inputs

## Getting Help

If you need help:

1. Check the README.md and documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join community discussions

## Recognition

Contributors will be recognized in:

- README.md contributors section
- CHANGELOG.md for significant contributions
- Release notes for major features

Thank you for contributing to the Fall Detection System for Elderly Care!