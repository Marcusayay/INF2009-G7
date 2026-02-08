// src/components/AlertDisclaim.jsx
import { HiExclamationTriangle, HiXMark, HiMinus } from "react-icons/hi2";
import { useState } from "react";
import { IoWarning } from "react-icons/io5";


export default function AlertDisclaim({ isVisible, setIsVisible }) {
  if (!isVisible) return null;

  return (
    <div className="bg-[#e9ecef] border-b border-gray-200 py-4 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto flex items-start justify-between">
        
        {/* Left Side: Icon & Content */}
        <div className="flex items-start space-x-4">
          <div className="">
            <IoWarning className="w-6 h-6 text-gray-700" />
          </div>
          
          <div className="flex flex-col">
            <h2 className="text-lg font-bold text-gray-800 leading-tight">
              Beware of impersonation scams
            </h2>
            <p className="mt-2 text-gray-600 text-base leading-relaxed max-w-4xl">
              Government officials will NEVER ask you to transfer money or disclose bank log-in details over a phone call. <br></br>
              Call the 24/7 ScamShield Helpline at 1799 or visit the{" "}
              <a 
                href="https://www.scamshield.gov.sg/" 
                target="_blank" 
                rel="noreferrer" 
                className="text-blue-700 underline font-medium hover:text-blue-900"
              >
                ScamShield website
              </a>{" "}
              if you are unsure if something is a scam.
            </p>
          </div>
        </div>

        {/* Right Side: Action Buttons */}
        <div className="flex items-center space-x-4 ml-4">
          <button className="text-gray-500 hover:text-gray-800 transition-colors">
            {/* <HiMinus className="w-6 h-6" /> */}
          </button>
          <button 
            onClick={() => setIsVisible(false)}
            className="text-gray-500 hover:text-gray-800 transition-colors"
          >
            <HiXMark className="w-6 h-6" />
          </button>
        </div>

      </div>
    </div>
  );
}