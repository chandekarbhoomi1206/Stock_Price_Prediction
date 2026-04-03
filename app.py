from flask import Flask, jsonify, render_template, request

from predictor import build_stock_response


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict")
def predict():
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "1y")
    target_date = request.args.get("target_date")
    try:
        payload = build_stock_response(ticker=ticker, period=period, target_date=target_date)
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify(
            {
                "error": "Something went wrong while fetching market data. Please try again."
            }
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
