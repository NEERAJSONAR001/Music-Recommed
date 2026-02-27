from flask import Flask, render_template_string, request
import subprocess
import sys
import os

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Test App</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            margin-top: 100px;
        }
        button {
            padding: 15px 30px;
            margin: 20px;
            font-size: 18px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>🎵 Model Testing App</h1>

    <form method="post">
        <button name="action" value="face">Test Face Model</button>
        <button name="action" value="text">Test Text Model</button>
    </form>

    <p>{{ message }}</p>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
    def home():
    message = ""

    if request.method == "POST":
    action = request.form["action"]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if action == "face":
            message = "Starting Face Model..."
            face_path = os.path.join(BASE_DIR, "face_emotions", "face_model.py")
            subprocess.Popen([sys.executable, face_path])

    elif action == "text":
            message = "Starting Text Model..."
            text_path = os.path.join(BASE_DIR, "text_emotions", "text_model.py")
            subprocess.Popen([sys.executable, text_path])


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template_string, request
import subprocess
import sys
import os

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Beats</title>
    <script src="https://cdn.tailwindcss.com"></script>

    <style>
        body {
            background: linear-gradient(135deg, #FFF4B3, #FFD86B);
            min-height: 100vh;
            transition: background 0.6s ease-in-out;
        }

        .glass {
            backdrop-filter: blur(20px);
            background: rgba(255,255,255,0.35);
            border-radius: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }

        .btn {
            border-radius: 16px;
            transition: all 0.3s ease;
        }

        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }

        .fade-in {
            animation: fadeIn 0.6s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px);}
            to { opacity: 1; transform: translateY(0);}
        }
    </style>
</head>

<body class="flex items-center justify-center">

<div class="w-full max-w-4xl p-8 text-center">

    <h1 class="text-5xl font-bold mb-4">🎧 Emotion Beats</h1>
    <p class="text-lg mb-10 text-gray-700">
        Music that understands you.
    </p>

    <div class="glass p-10">

        <form method="post" class="flex flex-col md:flex-row justify-center gap-6 mb-8">

            <button name="action" value="face"
                class="btn px-8 py-4 bg-black text-white font-semibold">
                📸 Scan Face
            </button>

            <button name="action" value="text"
                class="btn px-8 py-4 bg-white text-black font-semibold border">
                💬 Analyze Text
            </button>

        </form>

        {% if message %}
        <div class="fade-in">

            <h2 class="text-2xl font-semibold mb-4">
                {{ message }}
            </h2>

            <div class="mt-6">
                <p class="text-xl mb-2">
                    Emotion:
                    <span class="font-bold capitalize text-black">
                        {{ emotion }}
                    </span>
                </p>

                <div class="w-full bg-gray-300 rounded-full h-4 mt-4">
                    <div class="bg-black h-4 rounded-full transition-all duration-700"
                        style="width: {{ confidence }}%">
                    </div>
                </div>

                <p class="mt-2 text-gray-700">
                    {{ confidence }}% Confidence
                </p>

            </div>

        </div>
        {% endif %}

    </div>

</div>

</body>
</html>
"""



if __name__ == "__main__":
    app.run(debug=True)