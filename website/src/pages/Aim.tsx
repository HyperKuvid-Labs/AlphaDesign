import { Target, Rocket, Zap, Trophy, Cog, Globe } from 'lucide-react';

const Aim = () => {
  const objectives = [
    {
      icon: Target,
      title: "Comprehensive Modular Design",
      description: "Develop a framework that enables automated generation and integration of all major F1 car components (front wing, rear wing, chassis, suspension, etc.) for cohesive vehicle assembly."
    },
    {
      icon: Rocket,
      title: "Advanced Aerodynamic Optimization", 
      description: "Apply hybrid AI techniques—genetic algorithms and reinforcement learning—to maximize aerodynamic efficiency (downforce, drag reduction) across all modules while maintaining regulatory compliance."
    },
    {
      icon: Zap,
      title: "Continuous Action Space Control",
      description: "Implement RL algorithms like DDPG to allow real-valued, nuanced adjustments to design parameters, enabling finer control and more optimal solutions for aerodynamic features."
    },
    {
      icon: Trophy,
      title: "Efficient Computational Workflow",
      description: "Integrate intelligent CFD scheduling and resource management to reduce simulation overhead, accelerating the design cycle and enabling rapid prototyping."
    },
    {
      icon: Cog,
      title: "Physics-Informed Learning and Evaluation",
      description: "Use physics-based reward structures and Bezier curve parameterization to guide the RL agent toward physically plausible, manufacturable, and high-performance designs."
    },
    {
      icon: Globe,
      title: "Regulatory Compliance and Manufacturability",
      description: "Ensure all generated designs strictly adhere to FIA technical regulations and are feasible for real-world manufacturing, supporting practical application in motorsports."
    }
  ];

  const goals = [
    {
      category: "Short Term",
      items: [
        "Training on a heavier GPU's to get the best out of the models",
        "Close to real implementation of parts like the bezier curves"
      ]
    },
    {
      category: "Medium Term", 
      items: [
        "Change the entire empirical status of the project with proper test results to back it up",
        "Better fine grained stl libraries or get terminal access to a modern 3D softwares like fusion 360 or solidworks"
      ]
    },
    {
      category: "Long Term",
      items: [
        "Do this for a complete modular F1 car design",
        "Train in better gpus maybe tpus for longer time to get the best results"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-background py-20">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-20">
          <h1 className="text-4xl md:text-6xl font-rajdhani font-bold text-foreground mb-6">
            Project Aims & Vision
          </h1>
          <p className="text-xl font-inter text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Our mission is to redefine automotive design through precision engineering, 
            innovative technology, and unwavering commitment to excellence.
          </p>
        </div>

        {/* Core Objectives */}
        <div className="mb-20">
          <h2 className="text-3xl md:text-4xl font-rajdhani font-semibold text-center mb-12">
            Core Objectives
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {objectives.map((objective, index) => (
              <div 
                key={index}
                className="racing-card p-8 text-center group hover:border-primary/50"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-muted flex items-center justify-center transition-racing">
                  <objective.icon className="w-8 h-8 text-foreground" />
                </div>
                
                <h3 className="text-xl font-rajdhani font-semibold mb-4 text-foreground">
                  {objective.title}
                </h3>
                
                <p className="text-muted-foreground font-inter leading-relaxed">
                  {objective.description}
                </p>
              </div>
            ))}
          </div>
        </div>



        {/* Goals Timeline */}
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-rajdhani font-semibold text-center mb-12">
            Strategic Roadmap
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {goals.map((goalCategory, index) => (
              <div 
                key={index}
                className="racing-card p-8"
                style={{ animationDelay: `${index * 0.2}s` }}
              >
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-rajdhani font-semibold text-foreground mb-2">
                    {goalCategory.category}
                  </h3>
                </div>
                
                <ul className="space-y-4">
                  {goalCategory.items.map((item, itemIndex) => (
                    <li 
                      key={itemIndex}
                      className="flex items-start space-x-3 group"
                    >
                      <div className="w-2 h-2 rounded-full bg-muted-foreground mt-2 flex-shrink-0 transition-racing"></div>
                      <span className="text-muted-foreground font-inter leading-relaxed group-hover:text-foreground transition-racing">
                        {item}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>


      </div>
    </div>
  );
};

export default Aim;