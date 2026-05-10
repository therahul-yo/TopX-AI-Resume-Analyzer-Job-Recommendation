# 📚 A-Z Tech Stack & Foundation Interview Guide (Revature Prep)

This guide covers the fundamental "Must-Know" concepts for every technology used in your project. If the interviewer moves away from your project and starts asking general technical questions, this is your cheat sheet.

---

## 🐍 1. PYTHON (The Core)

**Q: What is the difference between a List and a Tuple?**
> **A:** Lists are **mutable** (you can change them after creation) and use `[]`. Tuples are **immutable** (cannot be changed) and use `()`. Tuples are generally faster and safer for data that shouldn't change.

**Q: What are Decorators in Python?**
> **A:** A decorator is a function that takes another function and extends its behavior without explicitly modifying it. (Example: `@app.route` in Flask is a decorator).

**Q: How is memory managed in Python?**
> **A:** Python uses **Automatic Garbage Collection** and **Reference Counting**. When an object's reference count drops to zero (nothing is using it), the memory is automatically freed.

**Q: What is the difference between `deepcopy` and `shallow copy`?**
> **A:** A **shallow copy** creates a new object but fills it with references to the original nested objects. A **deep copy** creates a new object and recursively creates copies of all nested objects.

---

## 🌶️ 2. FLASK (The Backend)

**Q: What is a "Microframework"?**
> **A:** It means Flask does not require particular tools or libraries. It doesn't have a built-in database layer or form validation. It gives you the "skeleton" (routing/requests) and lets you choose your own tools (like using `pdfminer` or `sqlite3`).

**Q: What is Jinja2?**
> **A:** It is the templating engine for Flask. It allows you to write Python-like code (`{% for %}`, `{{ variable }}`) inside HTML files to make them dynamic.

**Q: Difference between `redirect()` and `render_template()`?**
> **A:** `render_template()` shows a specific HTML file. `redirect()` sends the user to a different URL/Route entirely.

---

## 🗄️ 3. SQL & DATABASE (SQLite)

**Q: What are the ACID properties in a database?**
> **A:** 
> 1. **Atomicity:** All parts of a transaction succeed, or none do.
> 2. **Consistency:** Data remains in a valid state.
> 3. **Isolation:** Transactions don't interfere with each other.
> 4. **Durability:** Once saved, data stays saved even during a crash.

**Q: What is the difference between `DELETE` and `TRUNCATE`?**
> **A:** `DELETE` is a DML command that removes specific rows (can use `WHERE`). `TRUNCATE` is a DDL command that removes ALL rows from a table and is faster but cannot be rolled back easily in some DBs.

**Q: What is a "Join"?**
> **A:** A way to combine rows from two or more tables based on a related column between them. (Inner, Left, Right, Full).

---

## 🌐 4. HTML & CSS (The Frontend)

**Q: What is the "Box Model" in CSS?**
> **A:** Every element is a rectangular box consisting of: **Content** (text/images) -> **Padding** (space inside border) -> **Border** (line around padding) -> **Margin** (space outside border).

**Q: Difference between `display: none` and `visibility: hidden`?**
> **A:** `display: none` removes the element completely from the layout (it takes up no space). `visibility: hidden` hides the element but it still takes up the same space in the layout.

**Q: What is Responsive Design?**
> **A:** It's an approach where we use **Media Queries** to ensure a website looks good on all screen sizes (mobile, tablet, desktop).

---

## ⚡ 5. JAVASCRIPT (The Interactivity)

**Q: Difference between `var`, `let`, and `const`?**
> **A:** `var` is function-scoped (old way). `let` and `const` are block-scoped (modern way). `const` cannot be reassigned once set.

**Q: What is an "Event Loop" in JS?**
> **A:** It's what allows JS to perform non-blocking operations. It handles asynchronous tasks (like your Socket.IO messages) by pushing them to a queue and executing them when the main thread is free.

**Q: What is the DOM?**
> **A:** **Document Object Model**. It is a tree-like representation of your HTML that JavaScript uses to change text, colors, or add elements dynamically.

---

## 🤖 6. MACHINE LEARNING (The Intelligence)

**Q: What is Supervised vs Unsupervised Learning?**
> **A:** 
> *   **Supervised:** The model learns from labeled data (Input -> Correct Answer). Your project is Supervised because the model learned from a CSV where "Skills" were mapped to "Actual Companies."
> *   **Unsupervised:** The model finds patterns in unlabeled data (e.g., grouping customers by behavior).

**Q: What is Overfitting?**
> **A:** When a model learns the "noise" or specific details of the training data too well, failing to generalize to new, unseen data. (Analogy: Memorizing the textbook but failing the exam because the questions changed slightly).

**Q: What is a Random Forest?**
> **A:** It is an "Ensemble" model that consists of many **Decision Trees**. It takes the majority vote from all these trees to make a final prediction, making it more accurate and stable than a single tree.

---

## 🚀 7. FULL STACK FLOW (A-Z)

**If they ask: "Explain the A-Z flow of your project"**
> 1. **Frontend:** User interacts with the HTML/CSS/JS interface.
> 2. **Client-Side JS:** Captures the file and opens a **Socket.IO** connection.
> 3. **Web Server:** Gunicorn/Eventlet receives the request.
> 4. **Backend (Flask):** The route handler receives the PDF, triggers the **PDFMiner** parser.
> 5. **Processing:** Python scripts clean the text and extract features.
> 6. **Model (ML):** **Scikit-Learn** takes the features and predicts the company.
> 7. **Database:** SQLite handles user registration and login data.
> 8. **Feedback:** Socket.IO sends real-time progress back to the UI.
> 9. **Final Result:** Jinja2 renders the results page with all matched data.

---

## 💡 FINAL INTERVIEW ADVICE (The Revature Way)
*   **Don't panic if you don't know an answer:** Say, *"I haven't worked deeply with that specific concept yet, but in my project, I used [Related Concept] to solve [Problem]."*
*   **The "Full Stack" Mindset:** Always talk about how things *connect*. Don't just talk about Python; talk about how Python talks to the Database or the Frontend.
*   **Confidence:** Speak clearly. You built this. You know how it works. 

**YOU GOT THIS!** 🎓🔥
