import { useState } from "react";

export default function Toggle({ options = [], defaultValue, onChange }) {
  const [active, setActive] = useState(defaultValue ?? options[0]);

  const handleClick = (option) => {
    setActive(option);
    if (onChange) onChange(option);
  };

  return (
    <div className="flex justify-center mt-5 mb-12">
      <div className="bg-gray-200 p-1 rounded-full flex gap-1">
        {options.map((option) => (
          <button
            key={option}
            onClick={() => handleClick(option)}
            className={`px-8 py-1 text-sm font-medium rounded-full transition
              ${
                active === option
                  ? "bg-white shadow-sm"
                  : "text-gray-600"
              }`}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}
