from flask import Flask, render_template, request, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
model_id = "ibm-granite/granite-3.3-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cpu"},
        torch_dtype=torch.float32
    )
except Exception as e:
    print("Error loading Granite model:", e)
    exit()
feedback_list = []
app = Flask(__name__)
app.secret_key = "your_secret_key"
# ---------- Home Routes ----------
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/services")
def services():
    return render_template("services.html")
# ---------- Dashboard ----------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    sentiment_data = {"positive": 0, "neutral": 0, "negative": 0}
    for feedback in feedback_list:
        sentiment = analyze_sentiment(feedback)
        sentiment_data[sentiment.lower()] += 1
    return render_template("dashboard.html", username=session["username"], sentiment_data=sentiment_data)

# ---------- Login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "password":
            session["username"] = username
            return redirect(url_for("dashboard"))
        return "Invalid credentials", 401
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

# ---------- Chat Route ----------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        question = request.form.get("question")
        if not question:
            session["question_response"] = " No question provided."
        else:
            ai_response = granite_generate_response(question)
            session["question_response"] = ai_response
        return redirect(url_for("chat"))  # Redirect to avoid resubmission
    question_response = session.pop("question_response", None)
    if question_response is None:
        question_response = "Ask me anything!"
    return render_template("chat.html", username=session["username"], question_response=question_response)
# ---------- Feedback Route ----------
@app.route("/feedback", methods=["POST"])
def submit_feedback():
    feedback = request.form.get("feedback", "")
    feedback_list.append(feedback)  # Store for sentiment analysis
    sentiment = analyze_sentiment(feedback)
    return render_template("chat.html", username=session["username"], sentiment=sentiment)
# ---------- Concern Route ----------
@app.route("/concern", methods=["POST"])
def report_concern():
    concern_text = request.form.get("concern", "")
    print(f"Concern submitted: {concern_text}")
    return render_template("chat.html", username=session["username"], concern_submitted=True)
# ---------- Granite Response Generator ----------
def granite_generate_response(prompt):
    try:
        conv = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cpu")
        attention_mask = torch.ones_like(input_ids)
        set_seed(42)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print("Granite error:", e)
        return "Sorry, something went wrong while generating a response."
# ---------- Sentiment Analyzer ----------
def analyze_sentiment(text):
    if "good" in text.lower():
        return "Positive"
    elif "bad" in text.lower():
        return "Negative"
    return "Neutral"
# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True, port=2800)
