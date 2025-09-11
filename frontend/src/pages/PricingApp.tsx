import React, { useState } from "react";
import { UserRegistration } from "@/components/UserRegistration";
import { DynamicPricing } from "@/components/DynamicPricing";

interface UserData {
  user_type: "driver" | "rider";
  user_id: string;
  name: string;
  phone: string;
  city: string;
  area: string;
  rating: number;
  trips_completed: number;
}

const PricingApp: React.FC = () => {
  const [userData, setUserData] = useState<UserData | null>(null);

  const handleRegistrationComplete = (data: UserData) => {
    setUserData(data);
  };

  const handleBackToRegistration = () => {
    setUserData(null);
  };

  return (
    <div>
      {!userData ? (
        <UserRegistration onRegistrationComplete={handleRegistrationComplete} />
      ) : (
        <DynamicPricing userData={userData} onBack={handleBackToRegistration} />
      )}
    </div>
  );
};

export default PricingApp;
