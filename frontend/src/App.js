import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [studentData, setStudentData] = useState([]);

  useEffect(() => {
    axios.get('/api/student_performance')
      .then(response => {
        setStudentData(response.data);
      })
      .catch(error => {
        console.error('Error fetching student data:', error);
      });
  }, []);

  return (
    <div className="App">
      <h1>Student Performance Dashboard</h1>
      <table>
        <thead>
          <tr>
            <th>Student ID</th>
            <th>Activity</th>
            <th>Content ID</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {studentData.map((data, index) => (
            <tr key={index}>
              <td>{data.student_id}</td>
              <td>{data.activity}</td>
              <td>{data.content_id}</td>
              <td>{data.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
