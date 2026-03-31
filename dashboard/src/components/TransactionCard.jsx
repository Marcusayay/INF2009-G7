// components/TransactionCard.jsx
export default function TransactionCard({ type, material, weight, timestamp, icon }) {
  const typeText = (type || "").trim();
  const materialText = (material || "").trim().toLowerCase();
  const normalizedType = typeText.toLowerCase();
  const otherSubtypeMatch = typeText.match(/^others?\s*[\/:\-]\s*(.+)$/i);

  const toTitleCase = (value) =>
    value
      .replace(/[\/_-]+/g, " ")
      .trim()
      .replace(/\b\w/g, (char) => char.toUpperCase());

  const isGeneral = materialText === "general" || normalizedType === "general";
  const isKnownGeneralSubtype =
    normalizedType.includes("can") ||
    normalizedType.includes("bottle") ||
    normalizedType === "other" ||
    normalizedType === "others";

  let displayType = typeText;

  if (otherSubtypeMatch?.[1]) {
    displayType = `Others - ${toTitleCase(otherSubtypeMatch[1])}`;
  } else if (isGeneral && normalizedType === "general") {
    displayType = "General";
  } else if (isGeneral && !isKnownGeneralSubtype && typeText) {
    displayType = `General - Others (${typeText})`;
  } else if (isGeneral && (normalizedType === "other" || normalizedType === "others")) {
    displayType = "General - Others";
  }

  return (
    <div className="bg-white border border-gray-300 p-4 rounded-xl shadow-sm min-w-[180px]">
      <div className="mb-3 text-2xl">{icon}</div>
      <h4 className="font-bold text-gray-800">{displayType || "Unknown"}</h4>
      <p className="text-xs text-gray-500">{material} ({weight})</p>
      <p className="text-xs text-gray-400 mt-1">{timestamp}</p>
    </div>
  );
}