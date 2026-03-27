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
  const [brokerStatus, setBrokerStatus] = useState("connecting");
  const [brokerError, setBrokerError] = useState("");
  const [todayResetAt, setTodayResetAt] = useState(() => {
    const saved = localStorage.getItem("todayResetAt");
    const parsed = saved ? Number(saved) : 0;
    return Number.isFinite(parsed) ? parsed : 0;
  });

  const [data, setData] = useState({
    plastic: { bottles: 0 },
    metal: { cans: 0, bottles: 0 },
    glass: { bottles: 0 },
    general: { cans: 0, bottles: 0, others: 0 },
    tetra: { cartons: 0 }
  });

  const [transactions, setTransactions] = useState([]);

  const getTransactionTime = (tx) => {
    if (tx?.id) {
      const idTime = Number(tx.id);
      if (!Number.isNaN(idTime)) return idTime;
    }

    const dateMatch = (tx?.timestamp || "").match(/^(\d{2})\/(\d{2})\/(\d{4})/);
    if (!dateMatch) return null;

    const [, dd, mm, yyyy] = dateMatch;
    return new Date(Number(yyyy), Number(mm) - 1, Number(dd)).getTime();
  };

  const isTransactionFromToday = (tx) => {
    const now = new Date();
    const txTime = getTransactionTime(tx);

    if (todayResetAt > 0 && txTime !== null && txTime < todayResetAt) {
      return false;
    }

    if (todayResetAt > 0 && txTime === null) {
      return false;
    }

    if (tx?.id) {
      const idDate = new Date(Number(tx.id));
      if (!Number.isNaN(idDate.getTime())) {
        return idDate.toDateString() === now.toDateString();
      }
    }

    const dateMatch = (tx?.timestamp || "").match(/^(\d{2})\/(\d{2})\/(\d{4})/);
    if (!dateMatch) return false;

    const [, dd, mm, yyyy] = dateMatch;
    const parsed = new Date(Number(yyyy), Number(mm) - 1, Number(dd));
    return parsed.toDateString() === now.toDateString();
  };

  const buildCountsFromHistory = (historyItems) => {
    const counts = {
      plastic: { bottles: 0 },
      metal: { cans: 0, bottles: 0 },
      glass: { bottles: 0 },
      general: { cans: 0, bottles: 0, others: 0 },
      tetra: { cartons: 0 }
    };

    historyItems.forEach((item) => {
      const mat = item.material?.toLowerCase() || "";
      const type = item.type?.toLowerCase() || "";

      if (mat === "plastic" && type.includes("bottle")) counts.plastic.bottles += 1;
      if (mat === "glass" && type.includes("bottle")) counts.glass.bottles += 1;
      if (mat === "tetra" && type.includes("carton")) counts.tetra.cartons += 1;

      if (mat === "metal") {
        if (type.includes("can")) counts.metal.cans += 1;
        if (type.includes("bottle")) counts.metal.bottles += 1;
      }

      if (mat === "general") {
        if (type.includes("can")) counts.general.cans += 1;
        if (type.includes("bottle")) counts.general.bottles += 1;
        if (type.includes("other")) counts.general.others += 1;
      }
    });

    return counts;
  };

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
      setBrokerStatus("connected");
      setBrokerError("");
      console.log("Connected to MQTT Broker ✅");
      client.subscribe("pi/material/#");
    });

    client.on("reconnect", () => {
      setBrokerStatus("reconnecting");
    });

    client.on("offline", () => {
      setBrokerStatus("offline");
    });

    client.on("close", () => {
      setBrokerStatus("disconnected");
    });

    client.on("error", (err) => {
      setBrokerStatus("error");
      setBrokerError(err?.message || "Broker error");
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
                ...(material === "general" && { others: payload.totals.others ?? 0 })
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

  const resetTodayStats = () => {
    const resetTime = Date.now();
    setTodayResetAt(resetTime);
    localStorage.setItem("todayResetAt", String(resetTime));
  };

   const calculateCO2 = () => {
      const displayData = selectedRange === "Today"
        ? buildCountsFromHistory(transactions.filter(isTransactionFromToday))
        : data;

      const totalSaved =
        displayData.plastic.bottles * 0.05 +
        displayData.metal.cans * 0.09 +
        displayData.metal.bottles * 0.09 +
        displayData.glass.bottles * 0.10 +
        displayData.tetra.cartons * 0.08;

      // Return formatted to 2 decimal places
      return totalSaved.toFixed(2);
    };

  const displayTransactions = selectedRange === "Today"
    ? transactions.filter(isTransactionFromToday)
    : transactions;

  const displayData = selectedRange === "Today"
    ? buildCountsFromHistory(displayTransactions)
    : data;

  const brokerStatusUI = {
    connected: {
      label: "Connected",
      className: "border-emerald-200 bg-emerald-50 text-emerald-700"
    },
    connecting: {
      label: "Connecting",
      className: "border-amber-200 bg-amber-50 text-amber-700"
    },
    reconnecting: {
      label: "Reconnecting",
      className: "border-amber-200 bg-amber-50 text-amber-700"
    },
    offline: {
      label: "Offline",
      className: "border-orange-200 bg-orange-50 text-orange-700"
    },
    disconnected: {
      label: "Disconnected",
      className: "border-gray-200 bg-gray-100 text-gray-700"
    },
    error: {
      label: "Error",
      className: "border-red-200 bg-red-50 text-red-700"
    }
  }[brokerStatus] || {
    label: "Unknown",
    className: "border-gray-200 bg-gray-100 text-gray-700"
  };

  return (
    <div className="py-8 px-10 bg-[#f8fafd]">

      {/* Toggle Selector */}
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Toggle
          options={["Lifetime", "Today"]}
          defaultValue="Today"
          onChange={setSelectedRange}
        />
        <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
          <button
            type="button"
            onClick={resetTodayStats}
            className="w-full rounded-sm border border-red-300 bg-red-50 px-4 py-2 text-sm font-semibold text-red-700 transition hover:bg-red-100 sm:w-auto"
          >
            Reset Today Stats
          </button>
          <div className={`inline-flex items-center gap-2 rounded-sm border px-3 py-2 text-sm font-medium ${brokerStatusUI.className}`}>
            <span className="h-2 w-2 rounded-full bg-current opacity-80" />
            <span>Broker: {brokerStatusUI.label}</span>
          </div>
        </div>
      </div>

      {brokerStatus === "error" && brokerError ? (
        <p className="mb-4 text-sm text-red-700">Broker error: {brokerError}</p>
      ) : null}

      {/* Material Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 mb-6">
        <MaterialCard
          title="Plastic"
          cans={displayData.plastic.bottles}
          firstLabel="Bottles"
          showSecond={false}
        />
        <MaterialCard title="Metal" cans={displayData.metal.cans} bottles={displayData.metal.bottles} />
        <MaterialCard
          title="Glass"
          cans={displayData.glass.bottles}
          firstLabel="Bottles"
          showSecond={false}
        />
        <MaterialCard
          title="Tetra"
          cans={displayData.tetra.cartons}
          firstLabel="Cartons"
          showSecond={false}
        />
      </div>

      <div className="mb-8">
        <MaterialCard
          title="General"
          cans={displayData.general.cans}
          bottles={displayData.general.bottles}
          thirdLabel="Others"
          thirdValue={displayData.general.others}
          showThird={true}
        />
      </div>

      {/* Environmental Impact Banner */}
      <div className="bg-white border border-gray-200 p-4 rounded-lg flex justify-between items-center mb-10">
        <p className="text-blue-900 text-sm font-medium">
          This SCRAP bin has saved approximately <strong>{calculateCO2()}kg</strong> of CO2 since its operation.
        </p>
        <HiMiniSparkles className="text-blue-900" />
      </div>

      {/* Live Transactions */}
      <section>
        <h3 className="text-blue-900 font-semibold mb-4">Live Transactions</h3>
        <div className="bg-white border border-gray-200 p-6 rounded-sm mb-20 flex gap-4 overflow-x-auto min-h-[180px]">
          {displayTransactions.length === 0 ? (
            <p className="text-gray-400 italic">No recent activity...</p>
          ) : (
            displayTransactions.map((tx) => (
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