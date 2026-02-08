// components/TransactionCard.jsx
export default function TransactionCard({ type, material, weight, timestamp, icon }) {
  return (
    <div className="bg-white border border-gray-300 p-4 rounded-xl shadow-sm min-w-[180px]">
      <div className="mb-3 text-2xl">{icon}</div>
      <h4 className="font-bold text-gray-800">{type}</h4>
      <p className="text-xs text-gray-500">{material} ({weight})</p>
      <p className="text-xs text-gray-400 mt-1">{timestamp}</p>
    </div>
  );
}