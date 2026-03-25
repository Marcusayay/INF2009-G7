// layouts/BinItLayout.jsx
import React, { useState, useEffect } from "react"; // Added hooks here
import MaterialCard from "../components/MaterialCard";
import TransactionCard from "../components/TransactionCard";
import { FiStar } from "react-icons/fi"; // For the AI spark icon
import { HiMiniSparkles } from "react-icons/hi2";
import Toggle from "../components/Toggle";
import mqtt from "mqtt";

export default function HomePage() {
 
  const [selectedRange, setSelectedRange] = useState("Today");

  const [data, setData] = useState({
    plastic: { cans: 0, bottles: 0 },
    metal: { cans: 0, bottles: 0 },
    glass: { cans: 0, bottles: 0 },
    general: { cans: 0, bottles: 0 }
  });

  const [transactions, setTransactions] = useState([]);

  // Helper function moved outside useEffect or defined inside
  const getIcon = (mat, type) => {
    const material = mat?.toLowerCase() || "";
    const itemType = type?.toLowerCase() || "";

    if (material === "general") return "🍌";
    if (material === "glass") return "🗑️"; 
    if (itemType.includes("bottle")) return "🍼";
    if (itemType.includes("can")) return "🥫";
    return "📦"; 
  };

  useEffect(() => {
    const client = mqtt.connect("ws://localhost:9001");

    client.on("connect", () => {
      console.log("Connected to MQTT Broker ✅");
      client.subscribe("pi/material/#");
    });

    client.on("message", (topic, message) => {
      try {
        const material = topic.split("/").pop(); 
        const payload = JSON.parse(message.toString());

        // FIX 1: Accessing the new 'totals' object from your Python script
        if (payload.totals) {
          setData((prev) => ({
            ...prev,
            [material]: {
              cans: payload.totals.cans,
              bottles: payload.totals.bottles,
            },
          }));
        }

        // FIX 2: Use the 'history' array from Python to update the list
        // This ensures your "Cache" stays in sync even if you refresh
        if (payload.history) {
          const formattedHistory = payload.history.map(item => ({
            ...item,
            // We re-calculate icons based on the stored material/type
            icon: getIcon(item.material, item.type)
          }));
          setTransactions(formattedHistory);
        }

      } catch (err) {
        console.error("MQTT Message Error:", err);
      }
    });

    return () => client.end();
  }, []);

   const calculateCO2 = () => {
    // Constants (kg of CO2 saved per item)
      const factors = {
        plastic: { cans: 0.05, bottles: 0.05 }, // Adjust if plastic cans exist
        metal: { cans: 0.09, bottles: 0.09 },
        glass: { cans: 0.10, bottles: 0.10 },
        general: { cans: 0, bottles: 0 } // General waste usually doesn't save CO2
      };

      let totalSaved = 0;

      // Loop through each material in our state
      Object.keys(data).forEach((material) => {
        totalSaved += data[material].cans * factors[material].cans;
        totalSaved += data[material].bottles * factors[material].bottles;
      });

      // Return formatted to 2 decimal places
      return totalSaved.toFixed(2);
    };

  return (
    <div className="py-8 px-10 bg-[#f8fafd]">

      {/* Toggle Selector */}
      <Toggle
        options={["Lifetime", "Today"]}
        defaultValue="Today"
        onChange={setSelectedRange}
      />

      {/* Material Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 mb-8">
        <MaterialCard title="Plastic" cans={data.plastic.cans} bottles={data.plastic.bottles} />
        <MaterialCard title="Metal" cans={data.metal.cans} bottles={data.metal.bottles} />
        <MaterialCard title="Glass" cans={data.glass.cans} bottles={data.glass.bottles} />
        <MaterialCard title="General" cans={data.general.cans} bottles={data.general.bottles} />
      </div>

      {/* Environmental Impact Banner */}
      <div className="bg-white border border-gray-200 p-4 rounded-lg flex justify-between items-center mb-10">
        <p className="text-blue-900 text-sm font-medium">
          This BinIt has saved approximately <strong>{calculateCO2()}kg</strong> of CO2 since its operation.
        </p>
        <HiMiniSparkles className="text-blue-900" />
      </div>

      {/* Live Transactions */}
      <section>
        <h3 className="text-blue-900 font-semibold mb-4">Live Transactions</h3>
        <div className="bg-white border border-gray-200 p-6 rounded-sm mb-20 flex gap-4 overflow-x-auto min-h-[180px]">
          {transactions.length === 0 ? (
            <p className="text-gray-400 italic">No recent activity...</p>
          ) : (
            transactions.map((tx) => (
              <TransactionCard 
                key={tx.id}
                type={tx.type}
                material={tx.material}
                weight={tx.weight}
                timestamp={tx.timestamp}
                icon={tx.icon}
              />
            ))
          )}
        </div>
      </section>
    </div>
  );
}