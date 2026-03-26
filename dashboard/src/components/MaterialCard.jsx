// components/MaterialCard.jsx
export default function MaterialCard({
  title,
  cans,
  bottles,
  firstLabel = "Cans",
  secondLabel = "Bottles",
  showSecond = true,
  thirdLabel = "Others",
  thirdValue = 0,
  showThird = false
}) {
  return (
    <div className="bg-white border border-gray-200 p-6 rounded-sm shadow-sm">
      <h3 className="text-blue-900 font-semibold mb-4">{title}</h3>
      <div className={`grid ${showThird ? "grid-cols-3" : showSecond ? "grid-cols-2" : "grid-cols-1"} gap-4`}>
        <div className="flex flex-col">
          <span className="text-gray-600 text-sm mb-2">{firstLabel}</span>
          <div className="bg-[#3b4b72] text-white text-3xl font-bold py-4 text-center rounded-sm">
            {cans}
          </div>
        </div>
        {showSecond && (
          <div className="flex flex-col">
            <span className="text-gray-600 text-sm mb-2">{secondLabel}</span>
            <div className="bg-[#3b4b72] text-white text-3xl font-bold py-4 text-center rounded-sm">
              {bottles}
            </div>
          </div>
        )}
        {showThird && (
          <div className="flex flex-col">
            <span className="text-gray-600 text-sm mb-2">{thirdLabel}</span>
            <div className="bg-[#3b4b72] text-white text-3xl font-bold py-4 text-center rounded-sm">
              {thirdValue}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}