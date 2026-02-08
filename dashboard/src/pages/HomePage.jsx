// layouts/BinItLayout.jsx
import React from "react";
import MaterialCard from "../components/MaterialCard";
import TransactionCard from "../components/TransactionCard";
import { FiStar } from "react-icons/fi"; // For the AI spark icon
import { HiMiniSparkles } from "react-icons/hi2";
import Toggle from "../components/Toggle";

export default function HomePage() {
  const [selectedRange, setSelectedRange] = React.useState("Today");

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
        <MaterialCard title="Plastic" cans={973} bottles={80} />
        <MaterialCard title="Metal" cans={5} bottles={1} />
        <MaterialCard title="Glass" cans={0} bottles={0} />
        <MaterialCard title="General" cans={0} bottles={0} />
      </div>

      {/* Environmental Impact Banner */}
      <div className="bg-white border border-gray-200 p-4 rounded-lg flex justify-between items-center mb-10">
        <p className="text-blue-900 text-sm font-medium">
          This BinIt has saved over 200kg of CO2 since it's operation.
        </p>
        <HiMiniSparkles className="text-blue-900" />
      </div>

      {/* Live Transactions */}
      <section>
        <h3 className="text-blue-900 font-semibold mb-4">Live Transactions</h3>
        <div className="bg-white border border-gray-200 p-6 rounded-sm flex gap-4 overflow-x-auto">
          <TransactionCard 
            type="Bottle" 
            material="Plastic" 
            weight="15g" 
            timestamp="07/02/2026 | 23:06" 
            icon="🍼" 
          />
          <TransactionCard 
            type="Can" 
            material="Metal" 
            weight="15g" 
            timestamp="07/02/2026 | 22:56" 
            icon="🥫" 
          />
          <TransactionCard 
            type="Can" 
            material="General" 
            weight="250g" 
            timestamp="05/02/2026 | 12:31" 
            icon="🍌" 
          />
        </div>
      </section>
    </div>
  );
}