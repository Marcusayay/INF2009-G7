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
    plastic: { bottles: 0 },
    metal: { cans: 0, bottles: 0 },
    glass: { bottles: 0 },
    general: { cans: 0, bottles: 0 },
    tetra: { cartons: 0 }
  });

  const [transactions, setTransactions] = useState([]);

  // Helper function moved outside useEffect or defined inside
  const getIcon = (type) => {
    const itemType = type?.toLowerCase() || "";

    if (itemType.includes("carton")) return "🧃";
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
            ...(material === "plastic" && {
              plastic: { bottles: payload.totals.bottles ?? 0 }
            }),
            ...(material === "glass" && {
              glass: { bottles: payload.totals.bottles ?? 0 }
            }),
            ...(material === "tetra" && {
              tetra: { cartons: payload.totals.cartons ?? payload.totals.cans ?? 0 }
            }),
            ...(["metal", "general"].includes(material) && {
              [material]: {
                cans: payload.totals.cans ?? 0,
                bottles: payload.totals.bottles ?? 0,
              }
            })
          }));
        }

        // FIX 2: Use the 'history' array from Python to update the list
        // This ensures your "Cache" stays in sync even if you refresh
        if (payload.history) {
          const formattedHistory = payload.history.map(item => ({
            ...item,
            // We re-calculate icons based on the stored material/type
            icon: getIcon(item.type)
          }));

          const filteredHistory = formattedHistory.filter((item) => {
            const mat = item.material?.toLowerCase() || "";
            const type = item.type?.toLowerCase() || "";
            return !(type.includes("can") && (mat === "plastic" || mat === "glass"));
          });

          setTransactions(filteredHistory);
        }

      } catch (err) {
        console.error("MQTT Message Error:", err);
      }
    });

    return () => client.end();
  }, []);

   const calculateCO2 = () => {
      const totalSaved =
        data.plastic.bottles * 0.05 +
        data.metal.cans * 0.09 +
        data.metal.bottles * 0.09 +
        data.glass.bottles * 0.10 +
        data.tetra.cartons * 0.08;

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
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6 mb-8">
        <MaterialCard
          title="Plastic"
          cans={data.plastic.bottles}
          firstLabel="Bottles"
          showSecond={false}
        />
        <MaterialCard title="Metal" cans={data.metal.cans} bottles={data.metal.bottles} />
        <MaterialCard
          title="Glass"
          cans={data.glass.bottles}
          firstLabel="Bottles"
          showSecond={false}
        />
        <MaterialCard title="General" cans={data.general.cans} bottles={data.general.bottles} />
        <MaterialCard
          title="Tetra"
          cans={data.tetra.cartons}
          firstLabel="Cartons"
          showSecond={false}
        />
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