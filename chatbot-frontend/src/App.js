import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [chatVisible, setChatVisible] = useState(false);
  const [userMessage, setUserMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const chatboxRef = useRef(null);

  const toggleChatbot = () => {
    setChatVisible(!chatVisible);
  };

  const sendMessage = () => {
    if (userMessage.trim() === "") return;

    setChatHistory((prev) => [...prev, { sender: "You", text: userMessage }]);
    setUserMessage("");

    fetch("______/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data && data.response) {
          setChatHistory((prev) => [
            ...prev,
            { sender: "Noah", text: data.response },
          ]);
          chatboxRef.current.scrollTop = chatboxRef.current.scrollHeight;
        } else {
          throw new Error("Unexpected response format");
        }
      })
      .catch((error) => {
        console.error("Error during fetch:", error);
        setChatHistory((prev) => [
          ...prev,
          { sender: "Error", text: error.message },
        ]);
      });
  };

  const clearChat = () => {
    setChatHistory([]);
  };

  return (
    <div className="app">
      <h1>NLP Chatbot for Customer Support</h1>

      {/* Chatbot Button */}
      <button id="chatbot-button" onClick={toggleChatbot}>
        💬
      </button>

      {/* Chatbot Window */}
      {chatVisible && (
        <div id="chatbot-container">
          <div id="chatbot-header">
            <p
              style={{
                fontFamily: "Arial",
                margin: 0,
                fontWeight: "bold",
                marginLeft: "5px",
              }}
            >
              Customer Support Chatbot
            </p>
            <span id="close-btn" onClick={toggleChatbot}>
              ✖️
            </span>
          </div>

          <div id="chatbox" ref={chatboxRef}>
            {chatHistory.map((chat, index) => (
              <div key={index}>
                <strong>{chat.sender}:</strong> {chat.text}
              </div>
            ))}
          </div>

          <div style={{ padding: "10px" }}>
            <input
              id="user-input"
              type="text"
              placeholder="Type a message..."
              value={userMessage}
              onChange={(e) => setUserMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
            <button id="send-btn" onClick={sendMessage}>
              Send
            </button>
            <button id="clear-btn" onClick={clearChat}>
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
