import { Outlet, Link, useLocation } from "react-router-dom";
import AlertDisclaim from "../components/AlertDisclaim";
import { useState, useEffect } from "react";
import { RiRecycleFill } from "react-icons/ri";



export default function RootLayout() {
  const [isAlertVisible, setIsAlertVisible] = useState(true);
  const location = useLocation();

  const [currentDate, setCurrentDate] = useState("");
  const [currentTime, setCurrentTime] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();

      setCurrentDate(
        now.toLocaleDateString("en-SG", {
          weekday: "short",
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
        })
      );

      setCurrentTime(
        now.toLocaleTimeString("en-SG", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    };

    updateTime(); // initial render
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Use the light blue-grey background from the dashboard
  const mainBgClass = "bg-[#f8fafd]";

  return (
    <div className="flex flex-col min-h-screen bg-[#f8fafd]">
      {/* Header following the BinIt screenshot layout */}
      <header className="fixed w-full z-30 top-0 start-0">
        <nav className="bg-white border-b border-[#f0d9cf] w-full">
          <div className="max-w-screen flex items-center justify-between mx-auto p-4 sm:p-6">
            
            {/* Logo and Brand */}
            <div className="flex items-center gap-2">
              <RiRecycleFill size={28} color="#61A075" />
              <span className="text-2xl font-semibold text-[#3b4b72] tracking-tight">
                BinIt
              </span>
            </div>

            {/* Date Display */}
            <div className="text-right text-[#3b4b72]">
              <div className="text-xl font-semibold">{currentTime} | {currentDate}</div>
              {/* <div className="text-base font-semibold">{currentTime}</div> */}
            </div>
          </div>
        </nav>

        {/* Global Alert/Disclaimer */}
        {/* <AlertDisclaim isVisible={isAlertVisible} setIsVisible={setIsAlertVisible} /> */}
      </header>

      {/* Main Content Area */}
      <main className="pt-20 min-h-screen w-screen">
        {/* <div className="px-6 py-8"> */}
          <Outlet />
        {/* </div> */}
      </main>

      {/* Simplified Footer */}
      {/* <footer className="bg-white  fixed w-full z-30 bottom-0 start-0 border-t border-gray-200 p-8">
        <div className="mx-auto text-center">
          <p className="text-sm text-gray-400">
            © 2026 BinIt System. Data tracked for smart recycling.
          </p>
        </div>
      </footer> */}
    </div>
  );
}