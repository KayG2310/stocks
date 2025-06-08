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
  

// Start server
app.listen(3000, () => console.log("Server running on http://localhost:3000"));
