const express = require("express");
const session = require("express-session");
const path = require("path");
const bodyParser = require("body-parser");
const mongoose = require("mongoose");
const bcrypt = require('bcrypt');
const SALT_ROUNDS = 10;
const axios = require("axios");
const ALPHA_API_KEY = 'N9FWHMZE0CBTET7Z';
const stockSymbols = ["AAPL", "TSLA", "APPLE", "HDFCBANK.NS", "ICICIBANK.NS"];
const yahooFinance = require('yahoo-finance2').default;
const router = express.Router();
const { execSync } = require("child_process");
const { log } = require("console");

// POST route to handle search
router.post('/search', async (req, res) => {
  const query = req.body.query;

  try {
    // Step 1: Call your news API using the search query
    const response = await axios.get(`https://newsapi.org/v2/everything`, {
      params: {
        q: query,
        apiKey: process.env.NEWS_API_KEY,
      },
    });

    const articles = response.data.articles;

    // Step 2: Send this to your sentiment model (via another route or spawn Python)
    // Here’s a dummy example:
    // const sentiment = await analyzeSentiment(articles);

    // Step 3: Render results or redirect
    res.render('searchResult', { query, articles }); // or send to model
  } catch (error) {
    console.error('Search error:', error.message);
    res.status(500).send("Search failed");
  }
});

async function fetchStockData() {
    const stockSymbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS",
        "ADANIENT.NS", "ADANIGREEN.NS", "NTPC.NS", "MARUTI.NS", "M&M.NS"
      ];
      
  const stocks = [];

  for (const symbol of stockSymbols) {
    try {
      const quote = await yahooFinance.quote(symbol);
      stocks.push({
        symbol: quote.symbol.replace('.NS', ''),
        price: quote.regularMarketPrice,
        changePercent: (quote.regularMarketChangePercent * 100).toFixed(2) + '%',
      });
    } catch (err) {
      console.error(`Error fetching data for ${symbol}:`, err.message);
    }
  }

  return stocks;
}



mongoose.connect("mongodb://127.0.0.1:27017/stocknews", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("MongoDB Connected to stocknews"))
  .catch(err => console.log("MongoDB connection error:", err));
  const userSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true },
    password: String
  });
  
  const User = mongoose.model("User", userSchema);
  // This will use the "users" collection inside "stocknews" database
  
const app = express();
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.urlencoded({ extended: true }));

// Session config
app.use(session({
  secret: "secret-key",
  resave: false,
  saveUninitialized: false
}));

// In-memory user "DB"
const users = []; // Format: { name, email, password }

// Middleware to protect dashboard
function requireLogin(req, res, next) {
  if (!req.session.user) return res.redirect("/login");
  next();
}

// Routes
app.get("/", (req, res) => res.redirect("/dashboard"));

app.get("/signup", (req, res) => {
  res.render("signup", { error: null });
});

// Signup route
app.post("/signup", async (req, res) => {
    const { name, email, password } = req.body;
    try {
      const existing = await User.findOne({ email });
      if (existing) return res.render("signup", { error: "Email already registered." });
  
      const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
  
      const newUser = new User({ name, email, password: hashedPassword });
      await newUser.save();
  
      req.session.user = { name, email };
      res.redirect("/dashboard");
    } catch (err) {
      console.log(err);
      res.render("signup", { error: "Something went wrong." });
    }
  });
  
app.get("/login", (req, res) => {
  res.render("login", { error: null });
});

app.post("/login", async (req, res) => {
    const { email, password } = req.body;
    try {
      const user = await User.findOne({ email });
      if (!user) return res.render("login", { error: "Invalid credentials." });
  
      const match = await bcrypt.compare(password, user.password);
      if (!match) return res.render("login", { error: "Invalid credentials." });
  
      req.session.user = { name: user.name, email: user.email };
      res.redirect("/dashboard");
    } catch (err) {
      console.log(err);
      res.render("login", { error: "Something went wrong." });
    }
  });
  
app.get("/logout", (req, res) => {
  req.session.destroy();
  res.redirect("/login");
});
app.post("/logout", (req, res) => {
    req.session.destroy(err => {
      if (err) {
        console.log(err);
        return res.redirect("/dashboard"); // Or handle error page
      }
      res.clearCookie('connect.sid'); // Clears the session cookie
      res.redirect("/login");
    });
  });
  
  function isAuthenticated(req, res, next) {
    if (req.session && req.session.user) {
      return next();
    }
    res.redirect("/login");
  }
  
  app.get("/dashboard", isAuthenticated, async (req, res) => {
    const stocks = await fetchStockData();

    res.render("dashboard", {
        user: req.session.user,
        stocks: stocks,
    });
  });

app.get("/api/history/:symbol", async (req, res) => {
  const symbol = req.params.symbol || "NIFTYBEES.NS";

  try {
    const result = await yahooFinance.historical(symbol, {
      period1: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // last 7 days
      interval: '1d',
    });

    if (!result || result.length === 0) {
      return res.json([]);
    }

    const formattedData = result.map(d => ({
      time: d.date.toISOString().split("T")[0],
      price: d.open
    }));

    res.json(formattedData);
  } catch (err) {
    console.error("Yahoo Finance error:", err.message);
    res.status(500).json({ error: "Unable to fetch chart data." });
  }
});

app.post("/predict/:ticker", async (req, res) => {
  let ticker = req.params.ticker;
  console.log(`company is ${ticker}`);
  const venvPython = `"E:\\stocks\\backend\\python\\venv\\Scripts\\python.exe"`;
  try {
    //Step 1: Fetch news using fetch_news.py
    const newsRaw = execSync(`${venvPython} E:\\stocks\\backend\\python\\fetch_news.py "${company}"`).toString();
    const newsList = JSON.parse(newsRaw);
    const results = [];

    // Step 2: Analyze sentiment for each news headline
    sentimentScore = 0;
    for (const item of newsList) {
      const title = item.title.replace(/"/g, "'"); // Escape quotes
      let description = item.description.replace(/"/g, "'"); // Escape quotes
      description = description.replace("\n"," ");
      try{
        const output = execSync(`${venvPython} E:\\stocks\\backend\\python\\sentiment_analyzer.py "${title}: ${description}" "${company}"`).toString();

        const sentimentLine = output.split("\n")[0]
        const confidenceLine = output.split("\n")[1]

        sentiment = parseInt(sentimentLine?.split(":")[1]?.trim()) || 0;
        confidence = parseFloat(confidenceLine?.split(":")[1]?.trim()) || 0;

        sentimentScore += sentiment * confidence * 100;
      }
      catch(err){
        console.error("Pipeline error:", err.message);
      }
    }
    //Step 3: Predict
    try{
      const raw = execSync(`${venvPython} E:\\stocks\\backend\\python\\predict.py ${company}`).toString();
      const predicted = JSON.parse(raw); // This will be a JS array now
      let [open, high, low, close] = predicted;
      const changeFactor = (1 + sentimentScore*0.05);

      open = open*changeFactor;
      close = close*changeFactor;
      high = high*changeFactor;
      low = low*changeFactor;
    }
    catch(err){
      console.error("Pipeline error:", err.message);
    }

    res.json({
      ticker: company,
      results: sentimentScore,
      "open": open,
      "close": close,
      "high": high,
      "low": low
    });

  } catch (err) {
    console.error("Pipeline error:", err.message);
    res.status(500).json({ error: "Failed to run prediction pipeline." });
  }
});
  
  
  

// Start server
app.listen(3000, () => console.log("Server running on http://localhost:3000"));
