// components/MaterialCard.jsx
export default function MaterialCard({ title, cans, bottles }) {
  return (
    <div className="bg-white border border-gray-200 p-6 rounded-sm shadow-sm">
      <h3 className="text-blue-900 font-semibold mb-4">{title}</h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col">
          <span className="text-gray-600 text-sm mb-2">Cans</span>
          <div className="bg-[#3b4b72] text-white text-3xl font-bold py-4 text-center rounded-sm">
            {cans}
          </div>
        </div>
        <div className="flex flex-col">
          <span className="text-gray-600 text-sm mb-2">Bottles</span>
          <div className="bg-[#3b4b72] text-white text-3xl font-bold py-4 text-center rounded-sm">
            {bottles}
          </div>
        </div>
      </div>
    </div>
  );
}